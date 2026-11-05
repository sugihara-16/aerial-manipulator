#!/usr/bin/env python3
"""Generate a walking assembly, then independently walk the requested distance."""
import argparse,gc,json,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.walking import WalkSpec,foot_link
from amsrr_det.walking_plan import plant_for
from amsrr_det.walking_search import search,json_value
from amsrr_det.walking_control import GroundControl
from amsrr_det.walking_reference import KinematicReference
from amsrr_det.walking_loadpath import LoadPathReference
from amsrr_det.walking_audit import DriveAudit as WalkingAudit
from build.async_allocation import AsyncAllocation
from build.control_latency import ControlCycle


def prepare_execution(spec,plan,timestep,control_cpu,*,compiled_plant=None,preparation_cpus=None):
    if control_cpu is not None:os.sched_setaffinity(0,{control_cpu})
    plant=compiled_plant if compiled_plant is not None and compiled_plant.mj.opt.timestep==timestep else plant_for(plan,timestep)
    m=plant.mj
    gait={key:np.asarray(value) if isinstance(value,list) else value for key,value in plan['gait'].items()}
    links=[foot_link(*s) for s in plan['surfaces']];gait['foot_links']=links
    ref=KinematicReference(plant,spec,gait);q0=ref.pose(0)
    control=GroundControl(plant,q0,links,foot_points=gait.get('foot_points'));d=mujoco.MjData(m);d.qpos[:]=q0;mujoco.mj_forward(m,d)
    feet=control.positions(d)
    d.ctrl[:]=control.update(q0,np.zeros(m.nv),q0[:7],np.zeros(6),np.zeros(6),feet,
        np.zeros_like(feet),np.zeros_like(feet),np.ones(len(feet),bool),deadline=time.perf_counter()+1.)
    control.load_path=LoadPathReference(plant,control,ref,strength=.5,cpus=preparation_cpus)
    audit=WalkingAudit(plant,links,spec,ref)
    return plant,gait,links,ref,control,d,audit


def run(spec,out,*,seed=1,workers=24,worker_cpu=2,control_cpu=0,plan_file=None,timestep=.0005):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    from amsrr_det.valve_search import performance_cpus
    preparation_cpus=performance_cpus(set(os.sched_getaffinity(0)));execution_plant=[]
    # Leave measured setup time inside the same 15 s total acceptance bound.
    selection=search(spec,seed=seed,workers=workers,budget=14.6,execution_plant=execution_plant) if plan_file is None else json.loads(Path(plan_file).read_text())
    if plan_file is not None:spec=WalkSpec(**selection['request'])
    if plan_file is None and selection['status']=='accepted' and execution_plant:
        from amsrr_det.walking_refine import refine
        selection['plan'],selection['refinement']=refine(execution_plant[0],selection['plan'],spec,
            deadline=min(began+14.1,time.perf_counter()+4.),cpu=min(preparation_cpus) if preparation_cpus else None,cpus=preparation_cpus)
    (out/'request.json').write_text(json.dumps(asdict(spec),indent=2))
    (out/'selection.json').write_text(json.dumps(selection,default=json_value))
    if selection['status']!='accepted' or selection['plan'] is None:
        result=dict(task_success=False,reason=selection['status'],planning_s=selection['planning_s'])
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    if control_cpu is not None:os.sched_setaffinity(0,{control_cpu})
    plan=selection['plan']
    plant,gait,links,ref,control,d,audit=prepare_execution(spec,plan,timestep,control_cpu,compiled_plant=execution_plant[0] if execution_plant else None,preparation_cpus=preparation_cpus)
    m=plant.mj;allocation=AsyncAllocation(control,worker_cpu)
    planning=time.perf_counter()-began
    if plan_file is None and planning>15.:
        allocation.close();result=dict(task_success=False,reason='planning_timeout',planning_s=planning)
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    (out/'physical.xml').write_text(plant.xml);np.savez(out/'initial.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
    calls=[];states=[];poll_cpu=0.
    def plain(steps):
        for _ in range(steps):
            mujoco.mj_step(m,d);audit.physics(d)
            if audit.abort:return audit.abort
        return None
    def advance(steps):
        nonlocal poll_cpu
        end=float(d.time)+steps*m.opt.timestep;interval=max(1,round(.001/m.opt.timestep))
        while d.time<end-1e-10:
            remaining=max(0,round((end-d.time)/m.opt.timestep))
            if remaining<=interval:return plain(remaining)
            start=time.perf_counter();applied=allocation.poll(float(d.time))
            command=allocation.command.copy() if applied else None
            elapsed=time.perf_counter()-start;poll_cpu+=elapsed
            if applied:
                previous=d.ctrl.copy();delay=max(1,int(np.ceil(elapsed/m.opt.timestep-1e-10)))+1
                if plain(delay):return audit.abort
                stamp=time.perf_counter();d.ctrl[:]=command;elapsed+=time.perf_counter()-stamp
                extra=max(0,int(np.ceil(elapsed/m.opt.timestep-1e-10))-delay)
                if extra:
                    d.ctrl[:]=previous
                    if plain(extra):return audit.abort
                    d.ctrl[:]=command
                allocation.records[-1]['application_s']=float(d.time)
            n=min(max(0,round((end-d.time)/m.opt.timestep)),interval)
            if n and plain(n):return audit.abort
        return None
    gc.collect();gc.disable();started=time.perf_counter()
    try:
        audit.observe(d)
        while d.time<spec.time_limit-1e-9 and not audit.abort and not audit.reached():
            start=time.perf_counter();cycle=ControlCycle(m,d,.01);now=float(d.time);poll_cpu=0.
            q,v,a=ref.rates(now);feet,fv,fa,stance=ref.feet(now)
            args=(d.qpos.copy(),d.qvel.copy(),q[:7],v[:6],a[:6],feet,fv,fa,stance)
            cr,cv,ca=ref.com(now)
            kwargs=dict(shape_reference=q,shape_velocity=v,shape_acceleration=a,observation_time=now,
                        com_reference=cr,com_velocity=cv,com_acceleration=ca)
            command=allocation.update(now,'WALK',args,kwargs,observed_wall=start)
            elapsed=time.perf_counter()-start;cycle.wait_for_result(elapsed,plain)
            if not audit.abort:
                cycle.apply(command)
                if allocation.records and allocation.records[-1].get('application_tick_s')==cycle.observed_at and 'application_s' not in allocation.records[-1]:
                    allocation.records[-1]['application_s']=float(d.time)
                advance(cycle.remaining_steps())
            record=cycle.record('WALK');record['publication_cpu_s']=elapsed+poll_cpu;calls.append(record)
            audit.observe(d)
            if len(calls)%10==0:states.append(dict(time_s=float(d.time),q=d.qpos.tolist(),v=d.qvel.tolist(),ctrl=d.ctrl.tolist(),**control.last))
            if len(calls)%500==0:
                (out/'progress.json').write_text(json.dumps(audit.samples[-1]))
            if time.perf_counter()-started>600.:audit.abort='wall_time_limit'
    finally:
        allocation.close();gc.enable()
    result=audit.result();timing=np.array([r['publication_cpu_s'] for r in calls])*1000
    intervals=np.diff([r['observation_s'] for r in calls])
    result.update(modules=spec.modules,feet=len(links),planning_s=planning,fresh_structure_generation=plan_file is None,
        wall_s=time.perf_counter()-began,timestep=timestep,
        control=dict(calls=len(calls),mean_ms=float(timing.mean()),p99_ms=float(np.quantile(timing,.99)),max_ms=float(timing.max()),over10=int(sum(timing>=10))),
        allocation=dict(calls=len(allocation.records),errors=sum(r['error'] is not None for r in allocation.records),max_ms=max((r['calculation_s'] for r in allocation.records),default=0)*1000))
    result['control']['observation_over10']=int(np.count_nonzero(intervals>.01+1e-8))
    result['checks']['control_deadline']=result['control']['over10']==0 and result['control']['observation_over10']==0
    result['checks']['planning_deadline']=plan_file is not None or planning<=15.
    result['task_success']=all(result['checks'].values())
    for name,value in [('result',result),('calls',calls),('allocations',allocation.records),('states',states),('audit',audit.samples)]:
        (out/(name+'.json')).write_text(json.dumps(value,default=json_value))
    np.savez(out/'final.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--modules',type=int,default=4);ap.add_argument('--speed',type=float,default=.22);ap.add_argument('--curvature',type=float,default=0.);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--worker-cpu',type=int,default=2);ap.add_argument('--control-cpu',type=int,default=0);ap.add_argument('--plan');ap.add_argument('--timestep',type=float,default=.0005);ap.add_argument('--out',required=True);a=ap.parse_args()
    result=run(WalkSpec(modules=a.modules,speed=a.speed,curvature=a.curvature),a.out,seed=a.seed,workers=a.workers,worker_cpu=a.worker_cpu,control_cpu=a.control_cpu,plan_file=a.plan,timestep=a.timestep)
    print(json.dumps(result,default=json_value),flush=True)
