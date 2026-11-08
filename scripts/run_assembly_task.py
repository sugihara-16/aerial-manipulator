#!/usr/bin/env python3
"""Execute ground-start assembly with measured command latency and airflow."""
import argparse,gc,json,os,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
from amsrr_det.assembly_plan import prepare
from amsrr_det.assembly_central import CentralAssemblyControl,SmoothHandoff,seam_mask
from amsrr_det.assembly_fsm import AssemblyFSM
from amsrr_det.assembly_air import AssemblyAir,AirParameters
from build.async_allocation import AsyncAllocation
from build.control_latency import ControlCycle


def run(case,out,*,timestep=.0005,worker_cpu=2,air_strength=1.,observer=None,prepared_plan=None,return_context=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    (out/'input.json').write_text(json.dumps(case,indent=2))
    try:plan=prepare(case,timestep) if prepared_plan is None else prepared_plan
    except ValueError as error:
        result=dict(success=False,case=case['id'],modules=case['modules'],reason=str(error),planning_s=time.perf_counter()-begin)
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    p=plan['plant'];m=p.mj;d=mujoco.MjData(m);d.qpos[:]=plan['initial'];mujoco.mj_forward(m,d)
    fsm=AssemblyFSM(plan);air=AssemblyAir(p,case['initial_seed']+2,AirParameters(strength=air_strength))
    control=CentralAssemblyControl(plan);control.last_command[:]=0.
    handoff=SmoothHandoff(p,case['edges'])
    allocation=AsyncAllocation(control,worker_cpu)
    if observer is not None:observer.begin(m,p.xml)
    planning=time.perf_counter()-begin
    if planning>15.:
        allocation.close()
        result=dict(success=False,case=case['id'],modules=case['modules'],reason='planning_timeout',planning_s=planning)
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    (out/'physical.xml').write_text(p.xml)
    evidence=dict(plan['evidence'],rounds=plan['rounds'],air=asdict(air.parameters))
    (out/'plan.json').write_text(json.dumps(evidence,indent=2));np.savez(out/'planned.npz',initial=plan['initial'],closed=plan['closed'])
    calls=[];states=[];handoff_commands=[];abort=None;last_capture=-1.;poll_cpu=0.;peak_joint=0.;max_penetration=0.
    constraint_force=0.;constraint_moment=0.;effort_limits=np.array([a['effort']for a in p.meta['actuators']])
    base_force_indices=plan['vi'][:,None]+np.arange(3);base_moment_indices=plan['vi'][:,None]+np.arange(3,6)
    def record_handoff(before):
        if handoff.events and d.time-handoff.events[-1]['t']<.8:
            handoff_commands.append((float(d.time),before.copy(),d.ctrl.copy(),seam_mask(d.eq_active),allocation.last.get('central_mask',0)))
    def plain(count):
        nonlocal abort,peak_joint,max_penetration,constraint_force,constraint_moment
        for _ in range(count):
            air.apply(d);mujoco.mj_step(m,d)
            peak_joint=max(peak_joint,float(np.max(abs(d.actuator_force[:control.nj])/effort_limits)))
            if fsm.clock is not None:
                constraint_force=max(constraint_force,float(np.linalg.norm(d.qfrc_constraint[base_force_indices],axis=1).max()))
                constraint_moment=max(constraint_moment,float(np.linalg.norm(d.qfrc_constraint[base_moment_indices],axis=1).max()))
            if not np.isfinite(d.qpos).all() or np.max(abs(d.qvel))>200.:
                abort='extreme_instability';return abort
            if fsm.clock is not None and min(d.qpos[k+2]for k in plan['qi'])<.15:
                abort='fell';return abort
            for contact in d.contact:
                max_penetration=max(max_penetration,-float(contact.dist))
                if contact.dist<-.035:abort='extreme_collision';return abort
        return None
    def advance(count):
        nonlocal poll_cpu
        end=d.time+count*m.opt.timestep
        while d.time<end-1e-10:
            remaining=max(0,round((end-d.time)/m.opt.timestep))
            if remaining<=2:return plain(remaining)
            start=time.perf_counter();applied=allocation.poll(float(d.time))
            if applied and allocation.last.get('central_mask',0)!=seam_mask(d.eq_active):
                allocation.records[-1]['discarded_topology']=True;applied=False
            command=handoff.command(allocation.command,d.ctrl,float(d.time))if applied else None
            elapsed=time.perf_counter()-start;poll_cpu+=elapsed
            if applied:
                delay=max(1,int(np.ceil(elapsed/m.opt.timestep)))+1
                if plain(delay):return abort
                previous=d.ctrl.copy();stamp=time.perf_counter();d.ctrl[:]=command;elapsed+=time.perf_counter()-stamp
                extra=max(0,int(np.ceil(elapsed/m.opt.timestep))-delay)
                if extra:
                    d.ctrl[:]=previous
                    if plain(extra):return abort
                    d.ctrl[:]=command
                allocation.records[-1]['application_s']=float(d.time)
                record_handoff(previous)
            count=min(2,max(0,round((end-d.time)/m.opt.timestep)))
            if count and plain(count):return abort
    gc.collect();gc.disable();began=time.perf_counter();last_event=0;failure_hold_start=None;failure_hold=None
    if observer is not None:observer.ready()
    try:
        while d.time<46. and not fsm.success and not abort:
            start=time.perf_counter();cycle=ControlCycle(m,d,.01);now=float(d.time);poll_cpu=0.
            if failure_hold_start is None:
                fsm.control_ready=handoff.ready(now,allocation.last)
                reference,velocity,acceleration=fsm.update(d)
                if fsm.success:break
                if fsm.failure:
                    failure_hold_start=now;failure_hold=d.qpos.copy()
                    fsm.events.append(dict(t=now,event='failed_hold',reason=fsm.failure))
            if failure_hold_start is not None:
                if now-failure_hold_start>=1.:break
                reference=failure_hold;velocity=np.zeros(m.nv);acceleration=np.zeros(m.nv)
            handoff.note(d.eq_active,d.ctrl,now)
            args=(d.qpos.copy(),d.qvel.copy(),reference,velocity,acceleration,d.eq_active.copy())
            requested=allocation.update(now,fsm.stage,args,dict(observation_time=now,applied_command=d.ctrl.copy(),recovery=fsm.stage in ("BRAKE","RETREAT")),observed_wall=start)
            current_topology=allocation.last.get('central_mask',0)==seam_mask(d.eq_active)
            if not current_topology and allocation.records and allocation.records[-1].get('application_tick_s')==now:
                allocation.records[-1]['discarded_topology']=True
            command=handoff.command(requested,d.ctrl,now) if current_topology else d.ctrl.copy()
            if observer is not None:observer.update(d,fsm.stage)
            elapsed=time.perf_counter()-start;cycle.wait_for_result(elapsed,plain)
            if not abort:
                previous=d.ctrl.copy()
                cycle.apply(command)
                record_handoff(previous)
                if current_topology and allocation.records and allocation.records[-1].get('application_tick_s')==now and 'application_s' not in allocation.records[-1]:
                    allocation.records[-1]['application_s']=float(d.time)
                advance(cycle.remaining_steps())
            calls.append(dict(t=now,publication_s=elapsed+poll_cpu,phase=fsm.stage,allocation=allocation.last,
                              command_application_s=cycle.application_time,delay_position_change=cycle.position_change))
            if now-last_capture>=.1-1e-9:
                states.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.eq_active.copy(),fsm.stage));last_capture=now
            if len(fsm.events)>last_event:
                for event in fsm.events[last_event:]:print(json.dumps(event),flush=True)
                last_event=len(fsm.events)
    finally:
        boundary_wall=time.perf_counter()
        allocation.close();gc.enable()
    values=[x['publication_s']for x in calls]
    physical_success=bool(fsm.success and not fsm.failure and not abort);timing_ok=max(values,default=0.)<=.01
    task_end=float(d.time) if failure_hold_start is None else failure_hold_start
    result=dict(success=physical_success and timing_ok,physical_success=physical_success,
        reason=abort or fsm.failure or ('publication_deadline'if not timing_ok else ('assembled'if fsm.success else'execution_timeout')),
        case=case['id'],modules=case['modules'],planning_s=planning,simulation_s=float(d.time),assembly_s=None if fsm.clock is None else float(task_end-fsm.clock),
        hover_confirmed_s=fsm.clock,latched=int(d.eq_active.sum()),required=len(case['edges']),retries=fsm.retries,phase=fsm.stage,last_errors=fsm.errors,
        wall_s=time.perf_counter()-began,publication_max_s=max(values,default=0.),publication_over_10ms=sum(t>.01 for t in values),
        air_max_loss=air.maximum,air_loss_impulse_ns=air.integral,peak_joint_fraction=peak_joint,max_penetration_m=max_penetration,
        constraint_base_force_n=constraint_force,constraint_base_moment_nm=constraint_moment,
        failure_hold_started_s=failure_hold_start,post_failure_hold_s=0. if failure_hold_start is None else float(d.time-failure_hold_start),
        worker_max_s=max((x['calculation_s']for x in allocation.records),default=0.),worker_errors=[x for x in allocation.records if x['error']])
    result.update(central_mask=allocation.last.get('central_mask',0),leaders=allocation.last.get('leaders',{}),handoff_complete=handoff.ready(float(d.time),allocation.last))
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'events.json').write_text(json.dumps(fsm.events,indent=2))
    (out/'handoffs.json').write_text(json.dumps(handoff.events,indent=2))
    if handoff_commands:
        t,before,after,mask,authority=zip(*handoff_commands)
        np.savez_compressed(out/'handoff_commands.npz',t=t,before=before,after=after,mask=mask,authority=authority)
    if fsm.hover_observations:
        t,q,v,reference,ready=zip(*fsm.hover_observations)
        np.savez_compressed(out/'hover_observations.npz',t=t,q=q,v=v,reference=reference,ready=ready)
    (out/'timing.json').write_text(json.dumps(dict(calls=calls,allocation=allocation.records)))
    if states:
        t,q,v,u,e,phase=zip(*states);np.savez_compressed(out/'trajectory.npz',t=t,q=q,v=v,u=u,seams=e,phase=phase)
    np.savez(out/'final.npz',q=d.qpos,v=d.qvel,u=d.ctrl,seams=d.eq_active,time=d.time)
    if return_context:return result,dict(plan=plan,data=d,control=control,air=air,boundary_wall=boundary_wall)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('output',type=Path)
    parser.add_argument('--worker-cpu',type=int,default=2);parser.add_argument('--control-cpu',type=int,default=0)
    parser.add_argument('--timestep',type=float,default=.0005);parser.add_argument('--air-strength',type=float,default=1.)
    args=parser.parse_args();os.sched_setaffinity(0,{args.control_cpu})
    panel=json.loads(Path('configs/evaluation/assembly_generated.json').read_text())['cases']
    case=next(c for c in panel if c['id']==args.case)
    print(json.dumps(run(case,args.output,timestep=args.timestep,worker_cpu=args.worker_cpu,air_strength=args.air_strength)),flush=True)
