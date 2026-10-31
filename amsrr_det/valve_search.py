"""Count-wise valve synthesis with physical bounds and fixed-trial evidence."""
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .valve import ValveSpec,ValvePlant
from .valve_plan import candidate,complete_path
from .model import build_model
from .structure_bounds import count_lower_bounds,compare_count_groups
from .structure_stopping import required_trials,selection_evidence


def links_for(surfaces):
    return [f'm{n}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1]
            for n,port in surfaces]


def pack(plan):
    fields=['surfaces','closed','opened','empty_trim','final_open','final_trim','points','normals',
            'curve_record','angles','pin_path','geometry_angles','geometry_path','reference_path','intermediate_loads','direction_repairs','path_report','generation','geometry_samples','burden']
    result={k:getattr(plan,k) for k in fields}
    result.update(modules=plan.system.control.meta['modules'],edges=plan.system.control.meta['edges'])
    return result


def unpack(spec,record):
    system=ValvePlant(record['modules'],record['edges'],spec)
    system.configure_contacts(links_for(record['surfaces']))
    fields={k:v for k,v in record.items() if k not in ['modules','edges']}
    fields['surfaces']=list(map(tuple,fields['surfaces']))
    for k in ['closed','opened','final_open','points','normals','angles','pin_path','geometry_angles','geometry_path','reference_path']:
        if k in fields:fields[k]=np.asarray(fields[k])
    return SimpleNamespace(system=system,**fields)


def trial(payload):
    spec,n,index,seed=payload;started=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,index]))
    contacts=int(rng.integers(1,min(n,4)+1))
    try:
        plan=candidate(spec,n,contacts,rng)
        plan.system.configure_contacts(links_for(plan.surfaces))
        complete_path(plan)
        result=dict(status='accepted',burden=plan.burden,plan=pack(plan))
    except ValueError as error:
        result=dict(status='rejected',reason=str(error),burden=math.inf)
    result.update(modules=n,index=index,contacts=contacts,wall_s=time.perf_counter()-started)
    return result


_seed_affinity=None
_finish_affinity=None
_tail_affinity=None


def performance_cpus(cpus):
    """One logical CPU per fast physical core, within the allowed set."""
    if not cpus:return None
    try:
        root=Path('/sys/devices/system/cpu')
        frequencies={c:int((root/f'cpu{c}/cpufreq/cpuinfo_max_freq').read_text()) for c in cpus}
        fastest=max(frequencies.values());cores={}
        for c in cpus:
            if frequencies[c]<.9*fastest:continue
            topology=root/f'cpu{c}/topology'
            key=((topology/'physical_package_id').read_text(),(topology/'core_id').read_text())
            cores.setdefault(key,c)
        return set(cores.values())
    except (OSError,ValueError):return set(cpus)


def task_affinity(default):
    from . import planning_memo
    memo=planning_memo.active
    tail=_tail_affinity and memo is not None and (memo.directory/'fast_tail').exists()
    selected=_tail_affinity if tail else default
    if selected:os.sched_setaffinity(0,selected)


def seed_trial(payload):
    task_affinity(_seed_affinity)
    spec,n,index,seed=payload;started=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,index]))
    contacts=int(rng.integers(1,min(n,4)+1))
    from . import planning_memo
    if planning_memo.active is not None:planning_memo.active.task_group=n
    try:
        planning_memo.check_cancelled()
        plan=candidate(spec,n,contacts,rng,compile_model=False)
        fields=['closed','opened','empty_trim','surfaces','points','normals','fit','trim','generation','burden','pressure_reference']
        record={name:getattr(plan,name) for name in fields}
        record.update(modules=n,edges=plan.system.control.meta['edges'])
        result=dict(status='ready',seed_plan=record,burden=None)
    except planning_memo.PlanningCancelled:
        result=dict(status='cancelled',burden=None)
    except planning_memo.FixedOutputDominated as error:
        result=dict(status='bounded_nonimprovement',burden=error.evidence['lower_bound'],output_bound=error.evidence)
    except ValueError as error:
        result=dict(status='rejected',reason=str(error),burden=math.inf)
    result.update(modules=n,index=index,contacts=contacts,wall_s=time.perf_counter()-started)
    return result


def finish_trial(payload):
    task_affinity(_finish_affinity)
    spec,record=payload;started=time.perf_counter()
    from . import planning_memo
    if planning_memo.active is not None:planning_memo.active.task_group=record['modules']
    try:
        plan=unpack(spec,record['seed_plan']);complete_path(plan)
        result=dict(status='accepted',burden=plan.burden,plan=pack(plan))
    except planning_memo.PlanningCancelled:
        result=dict(status='cancelled',burden=None)
    except planning_memo.FixedOutputDominated as error:
        result=dict(status='bounded_nonimprovement',burden=error.evidence['lower_bound'],output_bound=error.evidence)
    except ValueError as error:
        result=dict(status='rejected',reason=str(error),burden=math.inf)
    result.update(modules=record['modules'],index=record['index'],contacts=record['contacts'],
        seed_s=record['wall_s'],wall_s=record['wall_s']+time.perf_counter()-started)
    return result


def _pin_worker(cpus,tail_cpus):
    global _seed_affinity,_finish_affinity,_tail_affinity
    _tail_affinity=tail_cpus
    if cpus:
        index=(mp.current_process()._identity[-1]-1)%len(cpus)
        _seed_affinity=_finish_affinity={cpus[index]};os.sched_setaffinity(0,_seed_affinity)


def search(spec,*,seed=1,workers=8,cpus=None,budget=15.,prepare=False,defer_cleanup=False):
    """Never count cancelled or unfinished draws as generator failures."""
    started=time.perf_counter();K=required_trials();_,meta=build_model()
    bounds=count_lower_bounds(meta['mass'],0.,9.80665)
    outcomes={n:[None]*K for n in range(1,9)};records=[];best=None
    # Include the fixed implementation and task in the evidence identifier.
    from .runtime_manifest import protocol_digest
    protocol=protocol_digest(spec.__dict__)
    from . import planning_memo
    # Workers share only this request's exact numerical evaluations. Prefer
    # volatile memory when available; cleanup still owns the whole directory.
    memory=Path('/dev/shm')
    memory_ok=memory.is_dir() and os.access(memory,os.W_OK) and os.statvfs(memory).f_bavail*os.statvfs(memory).f_frsize>256*1024**2
    cache=tempfile.TemporaryDirectory(prefix='valve_plan_',dir=memory if memory_ok else None)
    previous_memo=planning_memo.active;planning_memo.active=planning_memo.PlanningMemo(cache.name)
    # Immutable authored parts are shared by fork, rather than decoded and
    # hashed independently by every worker. No assembled candidate is stored.
    from .task_structure import part_frames
    from .contact_geometry import material_triangles
    from .mujoco_model import CONTACT_CACHE
    from .model import PORTS
    from .object_overlap import part_samples
    part_frames();part_frames(True)
    for port in PORTS:material_triangles(port,str(CONTACT_CACHE))
    part_samples()
    tail_cpus=performance_cpus(cpus);tail_started=False
    pool=ProcessPoolExecutor(workers,mp_context=mp.get_context('fork'),
        initializer=_pin_worker,initargs=(cpus,tail_cpus))
    pending={};ready=[];duplicates={};finished={};next_index={n:0 for n in range(1,9)};timed_out=False;prepared=None
    seed_times={n:[] for n in range(1,9)};timing={};deferred_cleanup=None
    def cleanup():
        began=time.perf_counter();cache.cleanup();timing['cleanup_s']=time.perf_counter()-began
    def eligible():
        if best is None:
            # Give every count an early batch. Difficult low-count solves
            # must not consume the budget before larger trees are proposed.
            return [n for n in range(1,9) if any(v is None for v in outcomes[n])]
        return [r['modules'] for r in compare_count_groups(bounds,best['burden']) if r['status']!='excluded_by_weight']
    try:
        while True:
            active=eligible()
            if best is not None:
                for n in set(range(1,9))-set(active):
                    (Path(cache.name)/('excluded_'+str(n))).touch(exist_ok=True)
            # Smallest physical lower bound first; finish every non-excluded
            # count's preassigned prefix. No first-success stop.
            ready=[row for row in ready if row['modules'] in active]
            if (tail_cpus and not tail_started and all(next_index[n]>=K for n in active)
                    and len(pending)+len(ready)<=len(tail_cpus)):
                # Near completion, fixed placement can leave one slow core
                # busy while fast cores are idle. Move only this request's
                # workers; numerical inputs and required draws do not change.
                (Path(cache.name)/'fast_tail').touch();tail_started=True
                timing['fast_tail_started_s']=time.perf_counter()-started
                for process in pool._processes.values():
                    try:os.sched_setaffinity(process.pid,tail_cpus)
                    except ProcessLookupError:pass
            for _ in range(workers-len(pending)):
                choices=[n for n in active if next_index[n]<K]
                finishing=sum(kind=='finish' for _,_,kind in pending.values())
                if ready and (finishing<max(1,workers//2) or not choices):
                    ready.sort(key=lambda row:(row['modules'],row['index']))
                    row=ready.pop(0);n,i=row['modules'],row['index']
                    future=pool.submit(finish_trial,(spec,row));pending[future]=(n,i,'finish')
                elif choices:
                    if best is not None and all(seed_times[n] for n in choices):
                        # Once an incumbent exists, launch groups with more
                        # estimated unfinished work first. Every group's same
                        # preassigned draws and physical exclusions remain.
                        n=max(choices,key=lambda n:((K-next_index[n])*np.mean(seed_times[n]),-n))
                    else:
                        n=min(choices,key=lambda n:(next_index[n] if best is not None else next_index[n]//4,bounds[n-1]['lower_bound']))
                    i=next_index[n];next_index[n]+=1
                    future=pool.submit(seed_trial,(spec,n,i,seed));pending[future]=(n,i,'seed')
                else:break
            if not pending:break
            remaining=budget-(time.perf_counter()-started)
            if remaining<=0:timed_out=True;break
            done,_=wait(pending,timeout=min(remaining,.1),return_when=FIRST_COMPLETED)
            for future in done:
                n,i,kind=pending.pop(future);row=future.result()
                if kind=='seed':seed_times[n].append(row['wall_s'])
                if row['status']=='ready':
                    source=row['seed_plan']
                    identity={key:source[key] for key in ['modules','edges','closed','surfaces','points','normals']}
                    key=hashlib.sha256(json.dumps(identity,sort_keys=True,default=json_value).encode()).hexdigest()
                    row['key']=key
                    if key in finished:
                        rows=[dict(finished[key],modules=n,index=i,contacts=row['contacts'],shared_evaluation=True)]
                    elif key in duplicates:
                        duplicates[key].append(row);continue
                    else:
                        duplicates[key]=[row];ready.append(row);continue
                elif kind=='finish':
                    key=next(key for key,rows in duplicates.items() if rows[0]['modules']==n and rows[0]['index']==i)
                    finished[key]=row
                    rows=[dict(row,modules=copy['modules'],index=copy['index'],contacts=copy['contacts'],shared_evaluation=j>0)
                          for j,copy in enumerate(duplicates.pop(key))]
                else:rows=[row]
                for row in rows:
                    n,i=row['modules'],row['index']
                    if row['status']=='cancelled':
                        records.append(row);continue
                    outcomes[n][i]=row['burden']
                    if row['status']=='accepted' and (best is None or row['burden']<best['burden']):
                        best=row
                        temporary=Path(cache.name)/'incumbent.tmp'
                        temporary.write_text(json.dumps(best['burden']))
                        temporary.replace(Path(cache.name)/'incumbent.json')
                    records.append({k:v for k,v in row.items() if k!='plan'})
            if best is not None and all(all(v is not None for v in outcomes[n]) for n in eligible()):break
        computation_finished=time.perf_counter()
        if prepare and best is not None and not timed_out and all(all(v is not None for v in outcomes[n]) for n in eligible()):
            # Restore the accepted numerical/physical model while this call's
            # exact-input cache exists. Include this work in planning_s.
            prepared=unpack(spec,best['plan'])
        preparation_finished=time.perf_counter()
    finally:
        # Only this request's worker pool is affected. Unfinished draws retain
        # None in the evidence; they are neither infinity nor successful.
        for future in pending:future.cancel()
        if pending:
            for process in pool._processes.values():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        shutdown_finished=time.perf_counter()
        planning_memo.active=previous_memo
        if defer_cleanup and prepared is not None:
            # No worker or model needs these files after preparation. Let the
            # caller dispose of them after it has started its control worker;
            # never start a cleanup thread before that caller's fork.
            deferred_cleanup=cleanup
        else:cleanup()
    censored={n:{r['index']:r['burden'] for r in records if r['modules']==n and r['status']=='bounded_nonimprovement'} for n in range(1,9)}
    evidence=None if best is None else selection_evidence(bounds,outcomes,best['burden'],protocol_id=protocol,censored_bounds=censored)
    complete=best is not None and evidence['complete'] and not timed_out
    timing.update(computation_s=computation_finished-started,
        preparation_s=preparation_finished-computation_finished,
        shutdown_s=shutdown_finished-preparation_finished,cleanup_deferred=deferred_cleanup is not None,
        fast_tail_cpus=sorted(tail_cpus) if tail_cpus else None)
    result=dict(status='accepted' if complete else 'planning_timeout' if timed_out else 'no_candidate',
                plan=None if best is None else best['plan'],evidence=evidence,trials=records,
                planning_s=time.perf_counter()-started,workers=workers,planning_cpus=cpus,seed=seed,
                timing=timing,
                request=spec.__dict__,
                criterion='maximum sampled static thrust burden, including free-flight ends')
    if prepare:result['_prepared_plan']=prepared
    if deferred_cleanup is not None:result['_cleanup']=deferred_cleanup
    return result


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
