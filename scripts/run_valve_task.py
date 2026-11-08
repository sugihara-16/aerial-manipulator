"""Execute a generated valve plan with measured-delay common allocation."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from amsrr_det.valve import ValveSpec
from amsrr_det.valve_search import unpack,search,json_value
from amsrr_det.fast_task_audit import FastTaskAudit
from build.relative_task_control import ObjectRelativeControl
from build.generated_grasp_stage import align_reference_to_object
from build.async_allocation import AsyncAllocation
from build.control_latency import ControlCycle
from build.contact_step_control import pad_gap_jacobians


class ValveReferences:
    def __init__(self,system,closed,opened,angles=None,path=None,final_open=None,surfaces=None,normals=None,geometry_angles=None,geometry_path=None,curve_record=None):
        self.system=system;self.spec=system.spec;self.m=system.control.mj
        self.closed=closed;self.opened=opened
        self.angles=angles;self.path=path;self.final_open=opened if final_open is None else final_open
        from amsrr_det.configuration_path import TabulatedContactPath,AxisRotationPath
        self.curve=None if path is None or curve_record is not None else TabulatedContactPath(system.control,angles,path,surfaces,normals,geometry_parameters=geometry_angles,geometry_configurations=geometry_path,object_shape=self.spec.shape)
        if curve_record is not None:self.curve=TabulatedContactPath.from_record(system.control,curve_record)
        # Rigid transport already preserves the screened material contacts.
        # Re-projecting all pads to one depth would change that very path,
        # including a discontinuous reference change when TURN begins.
        self.exact_material_motion=isinstance(self.curve,AxisRotationPath)
        self.delta=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,self.delta,1.,opened,closed)
        self.phases=['APPROACH','LOAD','TURN','SETTLE','UNLOAD','OPEN','RETREAT']
        self.durations=[4.,1.,10.,4.,3.,3.,3.]
        self.clock=None
        if path is not None:
            from amsrr_det.configuration_path import PathClock
            self.clock=PathClock(system.control,self.curve)
            self.durations[2]=self.clock.duration
        self.events=[];self.phase=None;self.complete=False;self.release=None
        self.gq=np.array([self.m.jnt_qposadr[self.m.joint(a['name']).id] for a in system.control.meta['actuators'] if a['role']=='vectoring'])
        self.gv=np.array([self.m.jnt_dofadr[self.m.joint(a['name']).id] for a in system.control.meta['actuators'] if a['role']=='vectoring'])

    def update(self,now,q,v,angle):
        index=min(int(np.searchsorted(np.cumsum(self.durations),now,side='right')),6)
        phase=self.phases[index];t=now-sum(self.durations[:index]);p=np.clip(t/self.durations[index],0,1)
        s=p**3*(10-15*p+6*p*p);ds=30*p*p*(1-p)**2/self.durations[index]
        dds=60*p*(1-p)*(1-2*p)/self.durations[index]**2
        if phase!=self.phase:self.events.append(dict(time_s=now,phase=phase));self.phase=phase
        self.complete=now>=sum(self.durations)+2.
        oi=self.system.coi;target=self.spec.pose();velocity=np.zeros(6);acc=np.zeros(6)
        shape=self.closed.copy();rate=np.zeros(self.m.nv);squeeze=1.;tracking=1.;motion=curvature=None
        theta=0.
        if phase=='APPROACH':
            shape=self.opened.copy();mujoco.mj_integratePos(self.m,shape,self.delta,float(s))
            rate=self.delta*ds;squeeze=s;tracking=0.
        elif phase=='TURN':
            theta,angular_speed,angular_acceleration=(self.clock.at(t) if self.clock is not None else
                (self.spec.target_angle*s,self.spec.target_angle*ds,self.spec.target_angle*dds))
            velocity[3:]=np.asarray(self.spec.axis)*angular_speed
            acc[3:]=np.asarray(self.spec.axis)*angular_acceleration
        elif phase in ['SETTLE','UNLOAD','OPEN','RETREAT']:theta=self.spec.target_angle
        target=self.spec.pose(theta)
        if self.path is not None and phase in ['TURN','SETTLE']:
            from build.momentum_feedback import rigid_motion_map
            shape,motion,curvature=self.curve.at(angle)
            speed=float(v[self.system.cov+3:self.system.cov+6]@np.array([0.,0.,1.]))
            rigid=rigid_motion_map(self.m,shape)@np.r_[np.zeros(3),np.asarray(self.spec.axis)*speed]
            rate=motion*speed-rigid
        if phase in ['UNLOAD','OPEN','RETREAT']:
            if self.release is None:
                self.release=(align_reference_to_object(self.m,self.closed if self.path is None else self.path[-1],q),align_reference_to_object(self.m,self.final_open,q))
                self.unloaded=self.release[0].copy();self.unloaded[self.gq]=self.release[1][self.gq]
                self.release_delta=np.zeros(self.m.nv)
                mujoco.mj_differentiatePos(self.m,self.release_delta,1.,self.unloaded,self.release[1])
            shape=self.unloaded.copy();mujoco.mj_integratePos(self.m,shape,self.release_delta,float(s if phase=='OPEN' else 1.))
            rate=self.release_delta*ds if phase=='OPEN' else rate
            target=shape[oi:oi+7].copy();squeeze=0.;tracking=0.
        bias=np.zeros(6)
        if phase in ['TURN','SETTLE']:
            torque=self.spec.torque*np.sign(self.spec.target_angle)+self.spec.damping*float(velocity[3:]@self.spec.axis)
            bias[3:]=np.asarray(self.spec.axis)*torque
        if phase=='UNLOAD':
            shape=self.release[0].copy();shape[self.gq]=(1-s)*self.release[0][self.gq]+s*self.unloaded[self.gq]
            rate=np.zeros(self.m.nv);rate[self.gv]=ds*(self.unloaded[self.gq]-self.release[0][self.gq])
            squeeze=1-s;tracking=0.;bias[3:]=np.asarray(self.spec.axis)*self.spec.torque*(1-s)
        if phase=='APPROACH':
            shape[self.gq]=self.opened[self.gq];rate[self.gv]=0.
        elif phase=='LOAD':
            shape[self.gq]=(1-s)*self.opened[self.gq]+s*self.closed[self.gq]
            rate[self.gv]=ds*(self.closed[self.gq]-self.opened[self.gq])
            bias[3:]=np.asarray(self.spec.axis)*self.spec.torque*s
        return (q,v,shape,target),dict(observation_time=now,object_velocity=velocity,object_acceleration=acc,shape_velocity=rate,
            payload_fraction=0.,tracking_fraction=tracking,squeeze=squeeze,wrench_bias=bias,
            track_object=phase not in ['OPEN','RETREAT'],
            shape_contact_depth=.0002 if phase in ['TURN','SETTLE'] and not self.exact_material_motion else None,
            motion_derivative=motion,motion_curvature=curvature)


def run(args,*,observer=None,plan_only=False,integration=None):
    out=args.output;out.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    spec=ValveSpec(**json.loads(args.request.read_text())) if args.request else ValveSpec()
    plan=None;cleanup=None
    if args.plan:
        selection=json.loads(args.plan.read_text())
        if not args.request:spec=ValveSpec(**selection['request'])
    else:
        selection=search(spec,seed=args.seed,workers=args.workers,cpus=args.planning_cpus,budget=14.7,prepare=True,defer_cleanup=True)
        plan=selection.pop('_prepared_plan')
        cleanup=selection.pop('_cleanup',None)
        (out/'selection.json').write_text(json.dumps(selection,default=json_value,indent=2))
        if selection['status']!='accepted':
            result=dict(task_success=False,reason=selection['status'],planning_s=time.perf_counter()-began)
            (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    if plan_only:
        if cleanup:cleanup()
        return dict(task_success=None,reason='structure_generated',planning_s=time.perf_counter()-began)
    if selection.get('plan') is None:raise ValueError('saved input has no complete candidate')
    if plan is None:plan=unpack(spec,selection['plan'])
    system=plan.system
    from amsrr_det.valve_contacts import require_side_contacts,require_side_geometry
    require_side_contacts(system.control,spec.shape,plan.closed,plan.surfaces)
    if args.plan:
        # Old saved plans may predate the side-only contract. Check their
        # full geometry before replay, without changing collision physics.
        query=mujoco.MjData(system.control.mj)
        for pose in plan.reference_path:
            query.qpos[:]=pose;mujoco.mj_fwdPosition(system.control.mj,query)
            require_side_geometry(system.control.mj,query,spec.shape)
    if args.plan:(out/'selection.json').write_text(json.dumps(selection,default=json_value))
    links=[f'm{n}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1] for n,port in plan.surfaces]
    plant=system.control;m=system.m
    angles=plan.angles;path=plan.reference_path;final_open=plant.configuration(plan.final_open)
    closed=path[0].copy()
    opened=plant.configuration(np.asarray(plan.opened))
    controller=ObjectRelativeControl(plant,closed,links,object_shape=spec.shape,
        task_projection=np.r_[np.zeros(3),spec.axis][None,:],vector_directions=True,material_friction_basis=True,free_flight_integral=True,object_face_axis=1,relative_pose_integral=True)
    controller.load.nominal_normal=max(.5,spec.torque/(.4*spec.length*len(links)))
    refs=ValveReferences(system,closed,opened,angles,path,final_open,plan.surfaces,plan.normals,
        plan.geometry_angles,plan.geometry_path,plan.curve_record)
    dr=controller.drive
    d=system.initialize(opened)
    d.ctrl[:dr.nj]=(dr.kp*opened[dr.qi]+np.array([j['effort_nm'] for j in plan.empty_trim['joints']]))/dr.gain[:dr.nj]
    d.ctrl[dr.nj:]=plan.empty_trim['thrust'];controller.last_command=d.ctrl.copy()
    mujoco.mj_forward(m,d)
    task_clock=0.;task_preparation=time.perf_counter()-began
    if integration is not None:
        physical_task=SimpleNamespace(mj=m,xml=system.xml,meta=plant.meta)
        physical,d=integration.start(physical_task,d);m=physical.mj;task_clock=float(d.time)
        system.m=m;system.xml=physical.xml
        system.oj=m.joint('object').id;system.oi=int(m.jnt_qposadr[system.oj]);system.ov=int(m.jnt_dofadr[system.oj])
        # Existing hinge adapter follows its reduced physical joint indices.
        system.pq=integration.map.pq[np.searchsorted(integration.map.cq,system.pq)]
        system.pv=integration.map.pv[np.searchsorted(integration.map.cv,system.pv)]
        from amsrr_det.task_sequence import BlendedControl
        controller=BlendedControl(controller,d.ctrl,integration.map)
    physical=SimpleNamespace(mj=m,meta=plant.meta)
    audit=FastTaskAudit(physical,links,spec.shape.size,0.,shape=spec.shape,clearance_period=.1,face_axis=1)
    allocation=AsyncAllocation(controller,args.worker_cpu)
    if observer is not None and integration is None:observer.begin(m,system.xml)
    planning=task_preparation if integration is not None else time.perf_counter()-began
    if args.plan is None and integration is None and planning>15.:
        allocation.close()
        if cleanup:cleanup()
        result=dict(task_success=False,reason='planning_timeout',planning_s=planning)
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    if cleanup:
        # Fork the allocator before starting any cleanup thread. File removal
        # is not needed for flight; its measured work overlaps real execution.
        import threading,os
        def housekeeping():
            if args.planning_cpus:os.sched_setaffinity(0,{args.planning_cpus[-1]})
            cleanup()
        threading.Thread(target=housekeeping,name='planning-housekeeping').start()
    from amsrr_det.valve_audit import ValveAudit
    physical_audit=ValveAudit(system)
    calls=[];trace=[];failure=None;poll_cpu=0.;applied_record=None;applied_solution={}
    (out/'physical.xml').write_text(system.xml);(out/'control.xml').write_text(plant.xml)
    (out/'request.json').write_text(json.dumps(spec.__dict__,indent=2))
    np.savez(out/'initial.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
    def plain(steps):
        nonlocal failure
        failure=integration.advance_audited(audit,steps) if integration is not None else audit.advance(d,steps)
        return failure
    def advance(steps):
        nonlocal poll_cpu,applied_record,applied_solution
        end=float(d.time)+steps*m.opt.timestep
        while d.time<end-1e-10:
            remaining=max(0,round((end-d.time)/m.opt.timestep))
            # The next observation has priority over the optional 1 kHz
            # receive check. Leave its final polling interval uninterrupted;
            # otherwise rounding a receive delay up can cross that boundary.
            if remaining<=max(1,round(.001/m.opt.timestep)):
                return plain(remaining)
            start=time.perf_counter();applied=allocation.poll(float(d.time))
            cmd=allocation.command.copy() if applied else None
            elapsed=time.perf_counter()-start;poll_cpu+=elapsed
            if applied:
                previous=d.ctrl.copy();delay_steps=max(1,int(np.ceil(elapsed/m.opt.timestep-1e-10)))+1
                if plain(delay_steps):return failure
                apply_start=time.perf_counter();d.ctrl[:]=cmd
                elapsed+=time.perf_counter()-apply_start
                extra=max(0,int(np.ceil(elapsed/m.opt.timestep-1e-10))-delay_steps)
                if extra:
                    d.ctrl[:]=previous
                    if plain(extra):return failure
                    d.ctrl[:]=cmd
                allocation.records[-1]['application_s']=float(d.time)
                applied_record=allocation.records[-1].copy();applied_solution=allocation.last.copy()
            n=min(max(0,round((end-d.time)/m.opt.timestep)),max(1,round(.001/m.opt.timestep)))
            if n and plain(n):return failure
        return failure
    gc.collect();gc.disable()
    if observer is not None:observer.ready()
    if integration is not None:
        integration.resume(plain);task_clock=float(d.time)
    try:
        while d.time-task_clock<(sum(refs.durations)+3.) and not refs.complete:
            start=time.perf_counter();cycle=ControlCycle(m,d,.01);poll_cpu=0.
            q,v=system.observation(d);a,k=refs.update(float(d.time)-task_clock,q,v,float(d.qpos[system.oi]))
            k['observation_time']=float(d.time)
            audit.face_active=refs.phase in ('LOAD','TURN','SETTLE','UNLOAD')
            command=allocation.update(float(d.time),refs.phase,a,k,observed_wall=start)
            if observer is not None:observer.update(d,refs.phase)
            elapsed=time.perf_counter()-start;cycle.wait_for_result(elapsed,plain)
            if not failure:
                cycle.apply(command)
                if allocation.records and allocation.records[-1].get('application_tick_s')==cycle.observed_at and 'application_s' not in allocation.records[-1]:
                    allocation.records[-1]['application_s']=float(d.time)
                    applied_record=allocation.records[-1].copy();applied_solution=allocation.last.copy()
                advance(cycle.remaining_steps())
            record=cycle.record(refs.phase);record['publication_cpu_s']=elapsed+poll_cpu;calls.append(record)
            if len(calls)%10==0:
                trace.append(dict(time_s=d.time,phase=refs.phase,angle=float(d.qpos[system.oi]),speed=float(d.qvel[system.ov]),**controller.last))
                physical_audit.record(d,refs.phase,applied_solution,applied_record,a[2])
            if len(calls)%500==0:(out/'progress.json').write_text(json.dumps(trace[-1]))
            if time.perf_counter()-began>600.:failure='wall_time_limit'
            if failure:break
    finally:allocation.close();gc.enable()
    query=mujoco.MjData(m);query.qpos[:]=d.qpos;mujoco.mj_fwdPosition(m,query)
    gaps,_=pad_gap_jacobians(m,query,links)
    checks=dict(angle=bool(abs(d.qpos[system.oi]-spec.target_angle)<np.deg2rad(5)),
        speed=bool(abs(d.qvel[system.ov])<.02),released=bool(len(gaps)>0 and min(gaps)>.02),
        settled=bool(audit.last_contact_time is not None and d.time-audit.last_contact_time>=2.))
    free_q,free_v=system.observation(d)
    retreat_error=np.zeros(plant.mj.nv)
    if refs.release is not None:
        mujoco.mj_differentiatePos(plant.mj,retreat_error,1.,free_q,refs.release[1])
    checks['retreat_flight']=bool(refs.release is not None and np.linalg.norm(retreat_error[:3])<.05
        and np.linalg.norm(retreat_error[3:6])<.2 and np.linalg.norm(free_v[:3])<.1 and np.linalg.norm(free_v[3:6])<.2)
    from amsrr_det.valve_audit import side_face_evidence
    side_faces=dict(audit.result()['contact_faces'],sampled=side_face_evidence(physical_audit.rows))
    checks['side_faces']=side_faces['passes']
    times=np.array([r['publication_cpu_s'] for r in calls])*1000
    result=dict(task_success=bool(not failure and all(checks.values())),reason=failure or ('completed' if all(checks.values()) else 'terminal_conditions_failed'),checks=checks,
        angle_rad=float(d.qpos[system.oi]),speed_rad_s=float(d.qvel[system.ov]),modules=plant.meta['modules'],contacts=len(links),
        planning_s=planning,task_start_s=task_clock,fresh_structure_generation=args.plan is None,wall_s=time.perf_counter()-began,simulation_s=float(d.time),audit=audit.result(),
        side_faces=side_faces,control=dict(calls=len(calls),mean_ms=float(times.mean()),p99_ms=float(np.quantile(times,.99)),max_ms=float(times.max()),over10=int(sum(times>=10))),
        allocation=dict(calls=len(allocation.records),errors=sum(r['error'] is not None for r in allocation.records),max_ms=max((r['calculation_s'] for r in allocation.records),default=0)*1000))
    (out/'selection.json').write_text(json.dumps(selection,default=json_value,indent=2))
    for name,value in [('result',result),('trace',trace),('control_calls',calls),('allocation_calls',allocation.records)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2))
    np.savez(out/'final.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl,retreat_goal=np.array([]) if refs.release is None else refs.release[1])
    physical_audit.save(out)
    (out/'events.json').write_text(json.dumps(refs.events,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--request',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--seed',type=int,default=1);parser.add_argument('--worker-cpu',type=int)
    parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--planning-cpus',type=lambda v:list(map(int,v.split(','))),default=list(range(32)))
    run(parser.parse_args())
