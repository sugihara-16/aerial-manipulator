"""Run one development task. The current structure is explicitly a seed."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import mujoco
from amsrr_det.objects import ObjectShape
from amsrr_det.grasp_task import plan_seed,GraspTask
from amsrr_det.fast_task_audit import FastTaskAudit as TaskAudit
from amsrr_det.evidence import save_sources


def run(shape,source,destination,support_top,output,timestep=.001,mode='yaw',axis_world=None,planner='generated'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    save_sources(output)
    request=dict(payload=shape.payload(),source=source.tolist(),destination=destination.tolist(),support_top=support_top,timestep=timestep,mode=mode,planner=planner,axis_world=None if axis_world is None else list(axis_world))
    (output/'request.json').write_text(json.dumps(request,indent=2))
    try:
        if planner=='generated':
            from amsrr_det.structure_search import plan_grasp,plan_three_contacts
            plan=(plan_grasp(shape,source,destination,support_top,preferred_mode=mode) if axis_world is None else
                  plan_three_contacts(shape,source,destination,support_top,mode=mode,axis_world=axis_world))
            plan.plant.mj.opt.timestep=timestep
            doc=ET.fromstring(plan.plant.xml)
            doc.find('option').set('timestep',str(timestep))
            plan.plant.xml=ET.tostring(doc,encoding='unicode')
        else:plan=plan_seed(shape,source,destination,support_top,timestep=timestep,mode=mode,axis_world=axis_world)
    except ValueError as e:
        if hasattr(e,'evidence'):(output/'planning_failure.json').write_text(json.dumps(e.evidence,indent=2))
        result=dict(task_success=False,reason='planning_failed:'+str(e),request=request)
        (output/'result.json').write_text(json.dumps(result,indent=2));return result
    return execute(plan,output,request)


def execute(plan,output,request, *, data=None, active_only=False, task_options=None, stop_at=None,reduced_physics=None,screen_prefix=False,physics_workers=0):
    """Independent full physics execution of an already generated candidate."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not (output/'source_snapshot').exists():save_sources(output)
    shape,source,destination,support_top=plan.shape,plan.source,plan.destination,plan.support_top
    timestep=plan.plant.mj.opt.timestep
    p=plan.plant;(output/'scene.xml').write_text(p.xml)
    (output/'plan.json').write_text(json.dumps(dict(scope='continuous_state_after_assembly' if data is not None else 'candidate_execution_no_assembly',
        initial_q=plan.initial_q.tolist(),open_q=plan.open_q.tolist(),closed_q=plan.closed_q.tolist(),
        surfaces=plan.surfaces,points=plan.points.tolist(),normals=plan.normals.tolist(),
        lift_height=plan.lift_height,approach_height=plan.approach_height,
        trims=plan.trims,fit=plan.fit,planning_time_s=plan.wall_s,model=p.meta,operating_point=plan.operating_point),indent=2))
    links=[f'm{m}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1] for m,port in plan.surfaces]
    full_model=p.mj
    if reduced_physics is None:reduced_physics=data is None
    if reduced_physics:
        if data is not None:raise ValueError('reduced evaluation currently requires completed-structure spawn')
        from amsrr_det.mujoco_model import contact_dynamics_model
        p=contact_dynamics_model(p,links)
        (output/'physics_scene.xml').write_text(p.xml)
    after_assembly=data is not None
    if data is None:
        data=mujoco.MjData(p.mj);data.qpos[:]=p.configuration(plan.initial_q);mujoco.mj_forward(p.mj,data)
    d=data;time_origin=float(d.time)
    if physics_workers<0:raise ValueError('physics_workers must be nonnegative')
    if physics_workers:mujoco.mju_threadpool(d,physics_workers)
    from amsrr_det.assembly import actuator_indices
    command_indices=actuator_indices(p)
    task=GraspTask(plan,**(task_options or {}));task.active_only=active_only
    if after_assembly:
        task.flight.command=d.ctrl[command_indices[:len(p.meta['actuators'])]].copy()
        task.flight.force=d.ctrl[command_indices[len(p.meta['actuators']):]].copy()
    audit=TaskAudit(p,links,shape.size,support_top,shape=shape,clearance_model=full_model)
    records=[];times=[];start=time.perf_counter();reason=None
    # Evaluation-only model output. No measured motor/contact value enters task.
    prefix=None;prefix_steps=0;prefix_squares=np.zeros(len(p.meta['actuators']))
    prefix_limits=np.array([a['effort'] for a in p.meta['actuators']])
    prefix_target=round(2./timestep)
    states=mujoco.mjtState.mjSTATE_INTEGRATION
    interval_state=np.empty(mujoco.mj_stateSize(p.mj,states))
    mujoco.mj_getState(p.mj,d,interval_state,states)
    issued_intervals=0
    step=0;step_limit=14000
    while step<step_limit:
        phase=task.stage;q,v=p.observation(d);before=time.perf_counter()
        try:cmd,forces=task.update(step*.01,q,v)
        except mujoco.FatalError as e:
            times.append(time.perf_counter()-before)
            reason='numerical_failure:controller_geometry:'+str(e);break
        times.append(time.perf_counter()-before)
        # A rejected prediction is not an input to execute for another period.
        if task.reason:
            reason=task.reason
            break
        # A qualified retreat has its own finite motion deadline. Do not cut
        # that path short solely because it extends beyond the old 140 s cap.
        # Pending planning still has the original outer cap; phase failures and
        # every physical/final audit remain unchanged.
        if task.stage=='RELEASE' and task.release_exit_timeout is not None:
            step_limit=max(step_limit,int(np.ceil((task.stage_start+task.release_exit_timeout)/.01))+2)
        d.ctrl[command_indices]=np.r_[cmd,forces]
        if phase!=task.stage or step%1000==0:
            print(json.dumps(dict(simulation_s=float(d.time)-time_origin,phase=task.stage,
                                  wall_s=time.perf_counter()-start)),flush=True)
        if phase!=task.stage:
            state=np.empty(mujoco.mj_stateSize(p.mj,states));mujoco.mj_getState(p.mj,d,state,states)
            np.savez_compressed(output/('checkpoint_'+task.stage+'.npz'),state=state,q=q,v=v,
                grasp_reference=task.grasp.reference if task.grasp else plan.closed_q,
                preload=task.grasp.preload if task.grasp else [],load=task.load,
                grasp_velocity=task.grasp.last_velocity if task.grasp else [],
                grasp_refresh=task.grasp.refresh_counter if task.grasp else 0,
                flight_trim=task.flight.trim_effort,
                force=task.flight.force,command=task.flight.command)
        # Retain the last issued, constant-input interval for exact numerical
        # replay. This evaluator state never enters the task/controller.
        mujoco.mj_getState(p.mj,d,interval_state,states)
        issued_intervals+=1
        for _ in range(round(.01/timestep)):
            try:mujoco.mj_step(p.mj,d)
            except mujoco.FatalError as e:reason='numerical_failure:'+str(e);break
            if any(d.warning[i].number for i in [mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC]):reason='numerical_failure';break
            try:audit.update(d,carried=phase=='CARRY',grasp_contact_allowed=phase!='APPROACH')
            except mujoco.FatalError as e:reason='numerical_failure:geometry_audit:'+str(e);break
            if audit.other_contacts:reason='unplanned_collision';break
            if audit.clearance_violations:reason='clearance_violation';break
            if audit.max_force>30.00001 or audit.max_moment>5.00001:reason='contact_load_limit';break
            if audit.max_penetration>.002:reason='penetration_limit';break
            if audit.max_joint_speed>3.00001 or audit.max_joint_limit_violation>1e-5:reason='joint_motion_limit';break
            if audit.longest_gap>.05:reason='carried_pad_load_gap';break
            if audit.max_material_displacement>.03:reason='contact_material_slip';break
            if audit.min_clearance_carried<.10:reason='transport_clearance';break
            if screen_prefix and prefix is None and phase=='CARRY' and not task.reason:
                prefix_squares+=(d.actuator_force[command_indices[:len(prefix_limits)]]/prefix_limits)**2
                prefix_steps+=1
                if prefix_steps==prefix_target:
                    means=prefix_squares/prefix_steps
                    prefix=dict(admissible=True,window_s=prefix_steps*timestep,
                        score=float(means.max()),mean_squared_normalized_efforts=means.tolist(),
                        actuator_names=[a['name'] for a in p.meta['actuators']],
                        simulation_s=float(d.time)-time_origin,wall_s=time.perf_counter()-start,
                        scope='continuous spawn through first two CARRY seconds; not full task acceptance')
                    (output/'prefix_screen.json').write_text(json.dumps(prefix,indent=2))
        if step%10==0:
            q,v=p.observation(d);oq=p.meta['object']['q']
            records.append(dict(t=float(d.time),phase=phase,configuration=q.tolist(),velocity=v.tolist(),object=q[oq:oq+7].tolist(),
                                grasp=task.grasp.last_diagnostics if task.grasp else None,
                                evaluator_normal_loads_n=audit.last_normal_loads.tolist(),actuator_commands=d.ctrl.tolist()))
        if reason or task.reason or task.finished:break
        if stop_at is not None and d.time-time_origin>=stop_at:
            reason='diagnostic_time_limit';break
        step+=1
    q,v=p.observation(d);oq,ov=p.meta['object']['q'],p.meta['object']['v'];final=q[oq:oq+7]
    if screen_prefix and prefix is None:
        prefix=dict(admissible=False,window_s=prefix_steps*timestep,reason=reason or task.reason or 'prefix_not_completed')
        (output/'prefix_screen.json').write_text(json.dumps(prefix,indent=2))
    state=np.empty(mujoco.mj_stateSize(p.mj,states));mujoco.mj_getState(p.mj,d,state,states)
    np.savez_compressed(output/'final_state.npz',state=state,qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl)
    np.savez_compressed(output/'last_issued_interval.npz',state=interval_state,
                        phase=phase,dt=timestep,duration_s=.01 if issued_intervals else 0.)
    if task.grasp and hasattr(task.grasp,'preload_failure_problem'):
        np.savez_compressed(output/'preload_failure_problem.npz',**task.grasp.preload_failure_problem)
    position_error=float(np.linalg.norm(final[:3]-destination[:3]));stats=audit.result()
    if not reason and not task.reason and not task.finished:reason='execution_horizon_exceeded'
    rotation_error=float(np.linalg.norm(pin.log3(pin.Quaternion(destination[3:]).matrix()@pin.Quaternion(final[3:]).matrix().T)))
    success=(task.finished and not reason and not task.reason and position_error<=.03 and rotation_error<=.20
             and stats['minimum_transport_clearance_m'] is not None and stats['minimum_transport_clearance_m']>=.10
             and stats['contact_material_displacement_m']<=.030 and stats['longest_carried_contact_gap_s']<=.05
             and stats['time_since_gripper_contact_s'] is not None and stats['time_since_gripper_contact_s']>=2.
             and np.linalg.norm(v[ov:ov+3])<=.02 and np.linalg.norm(v[ov+3:ov+6])<=.1)
    result=dict(pickup_reference=task.pickup_reference.tolist(),pickup_path_evidence=task.pickup_path_evidence,
                task_success=bool(success),reason=reason or task.reason or ('complete' if success else 'acceptance_failed'),
                scope='continuous_state_after_assembly' if after_assembly else 'prepared_structure_task_execution_only',phase=task.stage,phase_history=task.history,
                request=request,audit=stats,position_error_m=position_error,rotation_error_rad=rotation_error,final_object=final.tolist(),
                final_linear_speed_m_s=float(np.linalg.norm(v[ov:ov+3])),
                final_angular_speed_rad_s=float(np.linalg.norm(v[ov+3:ov+6])),controller_finished=task.finished,
                simulation_s=float(d.time)-time_origin,simulation_start_s=time_origin,wall_s=time.perf_counter()-start,planning_time_s=plan.wall_s,
                reduced_contact_dynamics=reduced_physics,full_geometry_audited=True,physics_workers=physics_workers,
                execution_horizon_s=step_limit*.01,execution_horizon_mode='prepared_retreat_deadline',
                prefix_screen=prefix,
                free_flight_switch=(dict(transitions=task.flight_switch.transitions,
                    last_forecast=task.flight_switch.last_forecast,
                    geometry_queries=task.flight_switch.clearance.queries,
                    geometry_reuses=task.flight_switch.clearance.reused)
                    if task.flight_switch is not None else None),
                failure_evidence=getattr(task,'failure_evidence',None),
                release_planning=(dict(waiting_simulation_s=task.supported_release.waiting,
                    wall_times_s=task.supported_release.planning_times,
                    path_ready=task.supported_release.retreat is not None)
                    if task.supported_release else None),
                control_time_s=dict(mean=float(np.mean(times)),p95=float(np.quantile(times,.95)),maximum=max(times)),
                final_reference_diagnostics=task.grasp.last_diagnostics if task.grasp else None)
    (output/'trace.json').write_text(json.dumps(records));(output/'result.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',default='box')
    parser.add_argument('--planner',choices=['generated','seed'],default='generated')
    parser.add_argument('--axis-angle',type=float,default=None)
    parser.add_argument('--mode',choices=['yaw','pitch'],default='yaw')
    parser.add_argument('--size',type=float,nargs=3,default=[.3,.4,.15]);parser.add_argument('--mass',type=float,default=1.)
    parser.add_argument('--yaw',type=float,default=0.);parser.add_argument('--roll',type=float,default=0.)
    parser.add_argument('--xy',type=float,nargs=2,default=[.745,0.]);parser.add_argument('--height',type=float,default=.175)
    parser.add_argument('--distance',type=float,default=.4);parser.add_argument('--timestep',type=float,default=.001)
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/task_box'));args=parser.parse_args()
    shape=ObjectShape(args.family,args.size,args.mass)
    quat=pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(args.roll),0.,np.deg2rad(args.yaw))).coeffs()
    source=np.r_[args.xy,0.,quat];source[2]=args.height-shape.lowest_z(source)
    destination=source.copy();destination[0]+=args.distance
    axis=None if args.axis_angle is None else [np.cos(np.deg2rad(args.axis_angle)),np.sin(np.deg2rad(args.axis_angle)),0.]
    result=run(shape,source,destination,args.height,args.output,args.timestep,args.mode,axis,args.planner)
    print(json.dumps({k:result[k] for k in ['task_success','reason','phase','simulation_s','wall_s','planning_time_s','position_error_m'] if k in result}),flush=True)


if __name__=='__main__':main()
