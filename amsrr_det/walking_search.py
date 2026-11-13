"""Fixed-module independent walking trials, compared by periodic burden."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
import multiprocessing as mp,time,os,signal
from .valve_search import performance_cpus
import numpy as np
from .walking_plan import candidate,generate,prepare_candidate,finish_candidate
from .structure_stopping import required_trials,fixed_count_evidence
from .planning_preview import SearchPreview,walking_candidate

_planning_cpus=None
_force_cpus=None
_planning_deadline=None
_active_forces=None
_tail_cpus=None
_geometry_priority_set=False

class _DeadlineExpired(TimeoutError):pass

def _expire(signum,frame):
    raise _DeadlineExpired()

def trial_stage(payload):
    """Cancel calculation between native calls, without corrupting result pipes."""
    remaining=_planning_deadline-time.perf_counter()
    if remaining<=0:return dict(status="rejected",reason="planning_deadline",cost=float("inf"),index=payload[2])
    signal.signal(signal.SIGALRM,_expire);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:return _trial_stage(payload)
    except _DeadlineExpired:return dict(status="rejected",reason="planning_deadline",cost=float("inf"),index=payload[2])
    finally:signal.setitimer(signal.ITIMER_REAL,0.)


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def trial(payload):
    spec,seed,index=payload;started=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([seed,spec.modules,index]))
    try:
        plan=candidate(spec,rng)
        result=dict(status='accepted',cost=plan['cost'],plan=plan,legs=len(plan['surfaces']))
    except ValueError as error:
        result=dict(status='rejected',reason=str(error),cost=float('inf'))
    result.update(index=index,wall_s=time.perf_counter()-started)
    return result


def _trial_stage(payload):
    global _geometry_priority_set
    spec,seed,index,stage,record=payload;started=time.perf_counter()
    previous_s=0. if record is None else record.get('preparation_s',record.get('_generation_s',0.))
    try:
        if stage=='generate':
            rng=np.random.default_rng(np.random.SeedSequence([seed,spec.modules,index]))
            record=generate(spec,rng);record['_generation_s']=time.perf_counter()-started
            return dict(status='generated',record=record,index=index)
        if stage=='prepare':
            # Change only this owned worker. Force calculations stay at the
            # original priority, so geometry work cannot delay their queue.
            if not _geometry_priority_set:
                os.nice(5);_geometry_priority_set=True
            os.sched_setaffinity(0,_planning_cpus)
            return dict(status='prepared',record=prepare_candidate(spec,None,record),index=index)
        os.sched_setaffinity(0,_force_cpus)
        with _active_forces.get_lock():_active_forces.value+=1
        try:
            # Use idle cores for independent phases at the tail of the queue.
            # The candidate, all checks and their result order are unchanged.
            def phase_cpus():
                with _active_forces.get_lock():active=_active_forces.value
                os.sched_setaffinity(0,_tail_cpus if active<=4 else _force_cpus)
                return _tail_cpus if active<=2 else None
            plan=finish_candidate(spec,record,phase_cpus=phase_cpus)
        finally:
            with _active_forces.get_lock():_active_forces.value-=1
        return dict(status='accepted',cost=plan['cost'],plan=plan,legs=len(plan['surfaces']),
                    index=index,wall_s=plan['wall_s'])
    except ValueError as error:
        return dict(status='rejected',reason=str(error),cost=float('inf'),index=index,
                    wall_s=previous_s+time.perf_counter()-started)


def search(spec,*,seed=1,workers=24,budget=15.,execution_plant=None):
    global _planning_cpus,_force_cpus,_planning_deadline,_active_forces,_tail_cpus
    preview=SearchPreview()
    _active_forces=mp.get_context('fork').Value('i',0)
    allowed=set(os.sched_getaffinity(0));fast_cpus=set(sorted(performance_cpus(allowed))[:8]);_force_cpus=allowed;_tail_cpus=fast_cpus
    # Geometry can use idle fast cores. Once force jobs become ready, their
    # higher process priority takes precedence without leaving cores reserved
    # for an empty queue. Every trial's numerical procedure stays unchanged.
    _planning_cpus=allowed;workers=min(workers,len(allowed))
    started=time.perf_counter();_planning_deadline=started+budget;count=required_trials(groups=1)
    from .runtime_manifest import protocol_digest
    from .task_structure import part_frames,port_axes
    from .mujoco_model import _part_templates,CONTACT_CACHE
    from .walking_native import library
    part_frames();part_frames(True);port_axes();_part_templates(str(CONTACT_CACHE.resolve()));library()
    from .walking_leaf import constants
    constants()
    from .mujoco_model import make_plant
    make_plant(1,mesh_cache=CONTACT_CACHE,floor=True,compile_threads=False)
    protocol=protocol_digest(asdict(spec));outcomes=[None]*count;records=[];best=None
    geometry=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'))
    forces=ProcessPoolExecutor(max_workers=min(workers,16),mp_context=mp.get_context('fork'))
    pending={geometry.submit(trial_stage,(spec,seed,i,'generate',None)):i for i in range(count)}
    generated={};foot_counts={};order=[];timed_out=False;generation_stage_s=None
    def complete(row):
        nonlocal best,timed_out
        if row.get("reason")=="planning_deadline":
            timed_out=True;records.append(row);return
        if row['status']=='accepted':
            plant=row['plan'].pop('_execution_plant',None)
            preview.outcome(spec.modules,'accepted',row['cost'],lambda:walking_candidate(row['plan'],plant))
            if execution_plant is not None and (best is None or row['cost']<best['cost']):
                execution_plant[:]=[plant]
        if row['status']!='accepted':preview.outcome(spec.modules,row['status'])
        outcomes[row['index']]=row['cost']
        if row['status']=='accepted' and (best is None or row['cost']<best['cost']):best=row
        records.append({key:value for key,value in row.items() if key!='plan'})
    def next_results():
        nonlocal timed_out
        remaining=budget-(time.perf_counter()-started)
        if remaining<=0:timed_out=True;return []
        done,_=wait(pending,timeout=min(.1,remaining),return_when=FIRST_COMPLETED)
        return done
    try:
        # A fast initial pose gives scheduling information for every seed.
        # Scheduling changes no trial's numerical procedure or outcome count.
        while pending and not timed_out:
            for future in next_results():
                pending.pop(future);row=future.result()
                if row['status']=='generated':
                    generated[row['index']]=row['record']
                    foot_counts[row['index']]=len(row['record']['surfaces'])
                else:complete(row)
        generation_stage_s=time.perf_counter()-started
        if not timed_out:
            # Schedule larger pose repairs first to reduce the long tail; all
            # fixed trials still complete and use their original computation.
            order=sorted(generated,key=lambda i:(-generated[i]['generation']['fit']['position_error_m'],i))
            next_index=0;active_geometry=set()
            while (pending or next_index<len(order)) and not timed_out:
                while next_index<len(order) and len(active_geometry)<workers:
                    i=order[next_index];next_index+=1
                    future=geometry.submit(trial_stage,(spec,seed,i,'prepare',generated.pop(i)))
                    pending[future]=i;active_geometry.add(future)
                for future in next_results():
                    i=pending.pop(future);active_geometry.discard(future);row=future.result()
                    if row['status']=='prepared':
                        pending[forces.submit(trial_stage,(spec,seed,i,'finish',row['record']))]=i
                    else:complete(row)
    finally:
        for future in pending:future.cancel()
        # Workers return a deadline result through an intact pipe. Killing a
        # worker while it writes a large plan can strand the result reader.
        for pool in [geometry,forces]:
            pool.shutdown(wait=True,cancel_futures=True)
    preview.finish()
    evidence=None if best is None else fixed_count_evidence(spec.modules,outcomes,best['cost'],protocol_id=protocol)
    return dict(status='planning_timeout' if timed_out else 'accepted' if best else 'no_candidate',
        request=asdict(spec),seed=seed,workers=workers,planning_s=time.perf_counter()-started,
        generation_stage_s=generation_stage_s,scheduling_order=order,
        fitted_proposal_foot_counts=foot_counts,
        evidence=evidence,trials=records,plan=None if best is None else best['plan'])
