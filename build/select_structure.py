"""Select by thrust burden, weight bounds and fixed independent trial prefixes.

The initial common seed is an extra incumbent, not an IID observation. Random
trials never receive that seed or other trial results. Scheduling can react to
the incumbent; the generator and each count's preassigned seed series cannot.
"""

import hashlib
import json
import secrets
from pathlib import Path
import time
import tempfile
import numpy as np
import mujoco
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
from amsrr_det.structure_search import grasp_candidates
from amsrr_det.structure_bounds import count_lower_bounds,compare_count_groups
from amsrr_det.structure_stopping import required_trials,selection_evidence
from amsrr_det.structure_workers import planning_context,TrialPool
from build.frozen_structure_trial import trial,prepare,screen_candidate,initialize_screen_memo


def initial_candidate(request,task_wrench=False):
    started=time.perf_counter();payload=request['payload']
    shape=ObjectShape(payload['shape'],payload['size'],payload['mass']);rows=[]
    try:
        plans=grasp_candidates(shape,np.asarray(request['source']),np.asarray(request['destination']),
                               request['support_top'],deadline=started+15.)
        for index,plan in enumerate(plans):
            try:
                prepared=prepare(plan,task_wrench=task_wrench)
                return dict(plan=plan,prepared=prepared,rows=rows,wall_s=time.perf_counter()-started)
            except (ValueError,mujoco.FatalError) as error:rows.append(dict(reason=str(error),modules=plan.plant.meta['modules']))
            if index>=7:break
    except (ValueError,mujoco.FatalError) as error:rows.append(dict(reason=str(error)))
    return dict(plan=None,prepared=None,rows=rows,wall_s=time.perf_counter()-started)


def task_initial_candidate(request,task_wrench,seed,starts=4):
    """Visit 1..8 modules with the exact main generator and common receiver.

    The auxiliary can change the count between candidates and return its first
    accepted candidate. Its outcomes never enter the per-count IID samples.
    """
    from build.frozen_structure_trial import generate_candidate
    started=time.perf_counter();rows=[]
    rng=np.random.default_rng(np.random.SeedSequence([seed,0]))
    for start in range(starts):
        for modules in range(1,9):
            candidate=generate_candidate(request,modules,rng,generator='task',common_initialization=True,retain_task_pose=True)
            row,plan,prepared=screen_candidate(request,candidate,task_wrench=task_wrench)
            row.update(start=start,modules=modules);rows.append(row)
            if prepared is not None:
                return dict(plan=plan,prepared=prepared,rows=rows,wall_s=time.perf_counter()-started,
                            status='completed',method='task_growth_variable_count')
    return dict(plan=None,prepared=None,rows=rows,wall_s=time.perf_counter()-started,
                status='completed_no_candidate',method='task_growth_variable_count')


def select_structure(request, *, workers=8,starts=4,seed=None,task_wrench=False,generator='task'):
    if generator not in ('legacy','task','mixed'):
        raise ValueError('unknown structure generator')
    if not isinstance(workers,int) or workers<1 or not isinstance(starts,int) or starts<1:
        raise ValueError('positive worker and per-trial start counts required')
    started=time.perf_counter();payload=request['payload']
    seed_source='fresh_system_entropy' if seed is None else 'explicit_replay'
    seed=secrets.randbits(128) if seed is None else seed
    if not isinstance(seed,int) or seed<0:raise ValueError('nonnegative integer seed required')
    _,meta=build_model(1,payload=ObjectShape(payload['shape'],payload['size'],payload['mass']).payload())
    bounds=count_lower_bounds(meta['mass']-meta['object']['mass'],meta['object']['mass'],9.80665,force_balance_tolerance=1e-6)
    count=required_trials();outcomes={n:[None]*count for n in range(1,9)}
    next_index={n:0 for n in outcomes};rows=[];best=None;prepared=None;initial=None
    root=Path(__file__).resolve().parents[1]
    paths=[*root.glob('amsrr_det/*.py'),*root.glob('build/*.py'),*root.glob('configs/robot/*.yaml'),
           root/'assets/robots/holon/holon.urdf',root/'requirements.lock.txt']
    paths.extend(root/p for p in ['CMakeLists.txt','native/geometry_kernel.cpp',
        'third_party/mujoco/contact.patch','third_party/mujoco/manifest.json',
        'build/geometry_kernel/libgeometry_kernel.so','build/geometry_kernel/identity.json',
        'build/body_bounds/python/mujoco/libmujoco.so.3.16.0',
        'assets/generated/authored_part_bounds.json','assets/generated/material_surfaces.npz'])
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    protocol=dict(sources=hashes,request=request,starts=starts,criterion='sum actual rotor thrust**1.5',
                  probability=.05,total_error=.05,groups=8,trials_per_group=count,task_wrench=task_wrench,
                  generator=generator,task_generation_probability={'legacy':0.,'task':1.,'mixed':.5}[generator],
                  auxiliary_initial_enabled=True,
                  auxiliary_generator='task_growth_variable_count' if generator=='task' else 'legacy_seed_growth',
                  calculation_reuse='exact-input memo within this planning call; draws and per-trial budgets unchanged',
                  scheduling='generate each trial in draw order; distribute its screens; collect only all completed starts',
                  generation_scope=('task contact roles; one role per module; variable-count auxiliary uses same task grammar'
                                    if generator=='task' else 'all legal free-pad subsets through legacy generation'),
                  task_contact_count_law='uniform 1..N per start, independent of nominal force feasibility',
                  task_contact_selection='normalized task-wrench residual, then force-boundary distance with 0.5 N normal floor',
                  task_pose_initialization='retain generated root, joints and contact targets for continuous surface fitting',
                  probability_scope='one more trial of this fixed generator; not existence of a better structure')
    protocol_id=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    def eligible():
        if prepared is None:
            return [n for n in outcomes if next_index[n]<count]
        return [r['modules'] for r in compare_count_groups(bounds,prepared['burden'])
                if r['status']=='unresolved' and next_index[r['modules']]<count]
    context=planning_context()
    memo=tempfile.TemporaryDirectory(prefix='amsrr_det_plan_',
        dir='/dev/shm' if Path('/dev/shm').is_dir() else None)
    pool=TrialPool(context,processes=workers,initializer=initialize_screen_memo,initargs=(memo.name,))
    pending={pool.apply_async(task_initial_candidate,(request,task_wrench,seed,starts)) if generator=='task'
             else pool.apply_async(initial_candidate,(request,task_wrench)):None}
    evidence=None;cancelled=[];cancelled_initial=False
    def collect(future):
        nonlocal initial,best,prepared
        key=pending.pop(future);result=future.get()
        result['collected_s']=time.perf_counter()-started
        plan=result.pop('plan');candidate=result.pop('prepared')
        if key is None:
            initial=dict(**result,burden=None if candidate is None else candidate['burden'])
        else:
            outcomes[key[0]][key[1]]=result['burden'];rows.append(result)
        if candidate is not None and (prepared is None or candidate['burden']<prepared['burden']):
            best,prepared=plan,candidate
    try:
        while True:
            # Drain complete results before deciding that work can be cancelled.
            for future in list(pending):
                if future.ready():collect(future)
            if prepared is not None:
                evidence=selection_evidence(bounds,outcomes,prepared['burden'],protocol_id=protocol_id)
                if evidence['complete']:
                    excluded={g['modules'] for g in evidence['groups'] if g['status']=='excluded_by_weight'}
                    if any(key is not None and key[0]not in excluded for key in pending.values()):
                        raise ValueError('refuse to cancel necessary independent trials')
                    cancelled=[dict(modules=n,trial=index,status='unfinished_physically_excluded')
                               for key in pending.values() if key is not None for n,index in [key]]
                    # The auxiliary seed is not an IID trial. Once every count
                    # has its evidence, awaiting it adds no required evidence.
                    cancelled_initial=initial is None
                    if cancelled_initial:
                        initial=dict(burden=None,rows=[],status='cancelled_after_required_evidence',
                                     completed=False,elapsed_s=time.perf_counter()-started)
                    break
            while len(pending)<workers:
                available=eligible()
                if not available:break
                n=min(available,key=lambda n:(bounds[n-1]['lower_bound'],n))
                index=next_index[n];next_index[n]+=1
                pending[pool.apply_async(trial,(request,n,index,seed),dict(starts=starts,task_wrench=task_wrench,
                    generator=generator,common_initialization=generator!='legacy',retain_task_pose=generator=='task'))]=(n,index)
            if not pending:break
            if not any(future.ready() for future in pending):time.sleep(.001)
    finally:
        # Public Pool shutdown stops only this selector's owned processes.
        # It does not leave irrelevant work competing with the flight control.
        pool.terminate();pool.join()
        memo.cleanup()
    if prepared is not None:evidence=selection_evidence(bounds,outcomes,prepared['burden'],protocol_id=protocol_id)
    report=dict(protocol=protocol,protocol_id=protocol_id,seed=seed,seed_source=seed_source,workers=workers,
                worker_start_method=context.get_start_method(),cancelled_excluded_trials=cancelled,
                cancelled_auxiliary_initial=cancelled_initial,
                wall_s=time.perf_counter()-started,initial=initial,rows=sorted(rows,key=lambda r:(r['modules'],r['trial'])),
                selection_evidence=evidence,selected_burden=prepared['burden'] if prepared is not None else None)
    if best is None or evidence is None or not evidence['complete']:
        error=ValueError('structure selection did not complete');error.evidence=report;raise error
    return best,prepared,report
