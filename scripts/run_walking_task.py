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


def prepare_execution(spec,plan,timestep,control_cpu,*,compiled_plant=None,preparation_cpus=None,air_start=False):
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
    control.load_path=LoadPathReference(plant,control,ref,cpus=preparation_cpus)
    if air_start:
        from amsrr_det.walking_landing import LandingReference,screen_landing
        if 'landing' not in plan:plan['landing']=screen_landing(plant,plan,spec)
        control.landing=LandingReference(plant,plan,ref)
        d.qpos[:]=control.landing.hover;d.qvel[:]=0.;d.ctrl[:]=control.landing.initial_command(control)
        mujoco.mj_forward(m,d)
    audit=WalkingAudit(plant,links,spec,ref,start_time=control.landing.duration if air_start else 0.)
    return plant,gait,links,ref,control,d,audit


def run(spec,out,*,seed=1,workers=24,worker_cpu=2,control_cpu=0,plan_file=None,timestep=.0005,air_start=True,observer=None,plan_only=False,integration=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    from amsrr_det.valve_search import performance_cpus
    preparation_cpus=performance_cpus(set(os.sched_getaffinity(0)));execution_plant=[]
    # Leave measured setup time inside the same 15 s total acceptance bound.
    selection=search(spec,seed=seed,workers=workers,budget=14.6,execution_plant=execution_plant) if plan_file is None else json.loads(Path(plan_file).read_text())
    if plan_file is not None:spec=WalkSpec(**selection['request'])
    if plan_file is None and selection['status']=='accepted' and execution_plant:
        from amsrr_det.walking_refine import refine
        selection['plan'],selection['refinement']=refine(execution_plant[0],selection['plan'],spec,
            # Continuous assembly needs additional setup before the first command.
            deadline=min(began+(12.2 if integration is not None else 14.1),time.perf_counter()+5.),cpu=min(preparation_cpus) if preparation_cpus else None,cpus=preparation_cpus)
    (out/'request.json').write_text(json.dumps(asdict(spec),indent=2))
    (out/'selection.json').write_text(json.dumps(selection,default=json_value))
    if selection['status']!='accepted' or selection['plan'] is None:
        result=dict(task_success=False,reason=selection['status'],planning_s=selection['planning_s'])
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    from amsrr_det.planning_preview import select_candidate,walking_candidate
    if execution_plant:select_candidate(lambda:walking_candidate(selection['plan'],execution_plant[0]))
    if plan_only:return dict(task_success=None,reason='structure_generated',planning_s=time.perf_counter()-began)
    if control_cpu is not None:os.sched_setaffinity(0,{control_cpu})
    plan=selection['plan']
    plant,gait,links,ref,control,d,audit=prepare_execution(spec,plan,timestep,control_cpu,compiled_plant=execution_plant[0] if execution_plant else None,preparation_cpus=preparation_cpus,air_start=air_start)
    landing=getattr(control,'landing',None);walk_start=landing.duration if landing else 0.
    task_clock=0.;physical=plant;task_preparation=time.perf_counter()-began
    if integration is not None:
        if landing:
            # This data is a desired pose, not the live assembly state.
            # Fly and morph directly to the descent entry posture.
            d.qpos[:]=landing.air_target;d.qvel[:]=0.
            landing.start_at_descent();walk_start=landing.duration
        physical,d=integration.start(plant,d);task_clock=float(d.time)
        audit=WalkingAudit(physical,links,spec,integration.reference(ref,plant),start_time=task_clock+walk_start)
        from amsrr_det.task_sequence import BlendedControl
        control=BlendedControl(control,d.ctrl,integration.map)
    whole=WalkingAudit(physical,links,spec,integration.reference(ref,plant) if integration is not None else ref,start_time=task_clock) if landing else None
    phases={};landing_load_time=0.;hover_gap=float("inf")
    m=physical.mj;allocation=AsyncAllocation(control,worker_cpu)
    if observer is not None and integration is None:observer.begin(m,plant.xml)
    planning=task_preparation if integration is not None else time.perf_counter()-began
    if plan_file is None and integration is None and planning>15.:
        allocation.close();result=dict(task_success=False,reason='planning_timeout',planning_s=planning)
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    (out/'physical.xml').write_text(physical.xml);np.savez(out/'initial.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
    calls=[];states=[];poll_cpu=0.
    def plain(steps):
        nonlocal landing_load_time,hover_gap
        for _ in range(steps):
            before=float(d.time)-task_clock
            if integration is not None:integration.air.apply(d)
            mujoco.mj_step(m,d)
            if whole and before<walk_start-1e-9:
                e=whole.energy_rotor;j=whole.energy_joint;w=whole.positive_work
                whole.physics(d)
                phase=landing.phase(before)
                entry=phases.setdefault(phase,dict(seconds=0.,rotor_proxy=0.,joint_proxy=0.,positive_joint_work_j=0.))
                entry['seconds']+=m.opt.timestep;entry['rotor_proxy']+=whole.energy_rotor-e
                entry['joint_proxy']+=whole.energy_joint-j;entry['positive_joint_work_j']+=whole.positive_work-w
                # Direct descent has no stationary HOLD; audit its actual
                # entry clearance over the first control period instead.
                if before<max(landing.hold_end,.01):
                    for bid,vertices in whole.material:
                        hover_gap=min(hover_gap,float((vertices@d.xmat[bid].reshape(3,3).T+d.xpos[bid])[:,2].min()))
                if walk_start-.3<=before<walk_start and whole.last_load.sum()>=.2*whole.weight:landing_load_time+=m.opt.timestep
                if whole.abort:audit.abort=whole.abort
            if before>=walk_start-1e-9:audit.physics(d)
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
    if observer is not None:observer.ready()
    if integration is not None:
        integration.resume(integration.advance_plain);task_clock=float(d.time)
        audit.start_time=task_clock+walk_start
        if whole:whole.start_time=task_clock
    try:
        if not landing:audit.observe(d)
        while d.time-task_clock<walk_start+spec.time_limit-1e-9 and not audit.abort and not audit.reached():
            start=time.perf_counter();cycle=ControlCycle(m,d,.01);now=float(d.time)-task_clock;poll_cpu=0.
            walking=now>=walk_start-1e-9
            if walking:
                if not audit.samples:audit.observe(d)
                local=max(0.,now-walk_start);q,v,a=ref.rates(local);feet,fv,fa,stance=ref.feet(local);phase='WALK';fraction=1.
                cr,cv,ca=ref.com(local)
            else:
                phase,q,v,a,feet,fv,fa,stance,fraction=landing.targets(now);cr=cv=ca=None
            observed_q,observed_v=integration.map.observe(d.qpos,d.qvel) if integration is not None else (d.qpos.copy(),d.qvel.copy())
            args=(observed_q,observed_v,q[:7],v[:6],a[:6],feet,fv,fa,stance)
            kwargs=dict(shape_reference=q,shape_velocity=v,shape_acceleration=a,observation_time=float(d.time),
                        com_reference=cr,com_velocity=cv,com_acceleration=ca,ground_fraction=fraction,walk_time=max(0.,now-walk_start),flight_mode=not walking)
            command=allocation.update(float(d.time),phase,args,kwargs,observed_wall=start)
            if observer is not None:observer.update(d,phase)
            elapsed=time.perf_counter()-start;cycle.wait_for_result(elapsed,plain)
            if not audit.abort:
                cycle.apply(command)
                if allocation.records and allocation.records[-1].get('application_tick_s')==cycle.observed_at and 'application_s' not in allocation.records[-1]:
                    allocation.records[-1]['application_s']=float(d.time)
                advance(cycle.remaining_steps())
            record=cycle.record(phase);record['publication_cpu_s']=elapsed+poll_cpu;calls.append(record)
            if walking:audit.observe(d)
            if len(calls)%10==0:states.append(dict(time_s=float(d.time),phase=phase,q=d.qpos.tolist(),v=d.qvel.tolist(),ctrl=d.ctrl.tolist(),**control.last))
            if len(calls)%500==0:
                (out/'progress.json').write_text(json.dumps(dict(phase=phase,time_s=float(d.time),walk=audit.samples[-1] if audit.samples else None)))
            if time.perf_counter()-started>600.:audit.abort='wall_time_limit'
    finally:
        allocation.close();gc.enable()
    result=audit.result();result.setdefault('checks',{'walking_executed':False})
    if whole:
        phases['WALK']=dict(seconds=audit.elapsed,rotor_proxy=audit.energy_rotor,joint_proxy=audit.energy_joint,positive_joint_work_j=audit.positive_work)
        # Whole-sequence totals are the disjoint prefix and walking integrals.
        whole.elapsed+=audit.elapsed;whole.energy_rotor+=audit.energy_rotor
        whole.energy_joint+=audit.energy_joint;whole.positive_work+=audit.positive_work
        whole.peak_penetration=max(whole.peak_penetration,audit.peak_penetration)
        whole.peak_effort=max(whole.peak_effort,audit.peak_effort)
        whole.abort=whole.abort or audit.abort
        result['checks']['airborne_landing']=landing_load_time>=.15 and hover_gap>.1 and whole.abort is None
        result['checks']['landing_face_alignment']=whole.face_angle_max<=np.deg2rad(5.) and whole.face_bad_impulse<=.01*max(whole.foot_impulse.sum(),1e-9)
        result['landing_face_alignment']=dict(maximum_loaded_angle_deg=float(np.rad2deg(whole.face_angle_max)),off_face_load_fraction=whole.face_bad_impulse/max(whole.foot_impulse.sum(),1e-9))
        result['whole_sequence']=dict(simulation_s=whole.elapsed,rotor_proxy=whole.energy_rotor,joint_proxy=whole.energy_joint,positive_joint_work_j=whole.positive_work,rotor_burden_hover_ratio=whole.energy_rotor/max(whole.elapsed*whole.weight**1.5/np.sqrt(len(whole.rotor_sites)),1e-9),phases=phases,walk_start_s=task_clock+walk_start,task_start_s=task_clock,landing_support_s=landing_load_time,hover_minimum_foot_gap_m=hover_gap,physical_abort=whole.abort,peak_penetration_m=whole.peak_penetration,peak_actuator_ratio=whole.peak_effort)
    timing=np.array([r['publication_cpu_s'] for r in calls])*1000
    intervals=np.diff([r['observation_s'] for r in calls])
    result.update(modules=spec.modules,feet=len(links),planning_s=planning,task_start_s=task_clock,fresh_structure_generation=plan_file is None,
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
    ap=argparse.ArgumentParser();ap.add_argument('--modules',type=int,default=4);ap.add_argument('--speed',type=float,default=.22);ap.add_argument('--curvature',type=float,default=0.);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--worker-cpu',type=int,default=2);ap.add_argument('--control-cpu',type=int,default=0);ap.add_argument('--plan');ap.add_argument('--ground-start',action='store_true');ap.add_argument('--timestep',type=float,default=.0005);ap.add_argument('--out',required=True);a=ap.parse_args()
    result=run(WalkSpec(modules=a.modules,speed=a.speed,curvature=a.curvature),a.out,seed=a.seed,workers=a.workers,worker_cpu=a.worker_cpu,control_cpu=a.control_cpu,plan_file=a.plan,timestep=a.timestep,air_start=not a.ground_start)
    print(json.dumps(result,default=json_value),flush=True)
