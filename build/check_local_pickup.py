"""Independent support-transfer and pickup diagnostic for the local design.

Starts with the computed grasp resting on the support. This is not acquisition
or complete task success. The controller sees only q/v; audit force truth is
kept outside it. References use static balance and acceleration endpoints,
never candidate trajectory acceptance.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
import pinocchio as pin
from build.load_saved_plan import load_saved_plan
from build.mit_drive import dock_mit_model
from build.sticking_design import LocalGrasp
from build.local_translation import LocalTranslation
from amsrr_det.mujoco_model import contact_dynamics_model
from amsrr_det.fast_task_audit import FastTaskAudit


def smooth(t):
    r=np.clip(t,0.,1.)
    return 10*r**3-15*r**4+6*r**5


def violation(audit):
    if audit.other_contacts:return 'unplanned_collision'
    if audit.clearance_violations:return 'clearance_violation'
    if audit.max_force>30.00001 or audit.max_moment>5.00001:return 'contact_load_limit'
    if audit.max_penetration>.002:return 'penetration_limit'
    if audit.max_joint_speed>3.00001 or audit.max_joint_limit_violation>1e-5:return 'joint_motion_limit'
    if audit.longest_gap>.05:return 'carried_pad_load_gap'
    if audit.max_material_displacement>.03:return 'contact_material_slip'
    if audit.min_clearance_carried<.10:return 'transport_clearance'
    return None


def prepare(local,qair,plan,displacement=None,*,predictive=False,motion_factory=None):
    m=local.m;qground=qair.copy()
    for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):qground[m.jnt_qposadr[j]+2]-=.15
    from build.support_contacts import support_contacts
    supports=support_contacts(m,qground)
    if not supports:raise ValueError('no geometric support at the pickup reference')
    supported_model=LocalGrasp(local.plant,qground,local.links,support=supports)
    supported,support_info=supported_model.equilibrium(qground)
    if not support_info['feasible']:raise ValueError('supported equilibrium unavailable')
    from build.local_translation import generated_translation
    motion=motion_factory or (generated_translation if predictive else LocalTranslation)
    lift=motion(local,qground,np.array([0.,0.,.15]),4.)
    carry=motion(local,qair,plan.destination[:3]-plan.source[:3]if displacement is None else displacement,6.)
    lower=np.r_[np.where(local.dock,local.kp*qair[local.qi]-local.effort,[a['lower'] for a in local.plant.meta['actuators']]),local.rotor_min]
    upper=np.r_[np.where(local.dock,local.kp*qair[local.qi]+local.effort,[a['upper'] for a in local.plant.meta['actuators']]),local.rotor_max]
    return qground,supported,support_info,lift,carry,lower,upper


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=int,default=2)
    parser.add_argument('--design',type=Path,default=Path('artifacts/runs/local_equilibrium_kp200_fixed'))
    parser.add_argument('--duration',type=float,default=16.)
    parser.add_argument('--start',choices=['held','open'],default='held')
    parser.add_argument('--flight',choices=['separate','coupled'],default='separate')
    parser.add_argument('--geometric-acquisition',action='store_true')
    parser.add_argument('--place',action='store_true')
    parser.add_argument('--initial-state',type=Path)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    began=time.perf_counter()
    plan,_=load_saved_plan(f'artifacts/runs/sticking_geometry_pair/candidate{args.candidate}')
    links=[f'm{m}/'+('yaw_dock_mech' if p.startswith('yaw') else 'pitch_dock_mech')+p[-1] for m,p in plan.surfaces]
    full=dock_mit_model(plan.plant,kp=200.)
    plant=contact_dynamics_model(full,links);m=plant.mj;m.opt.timestep=.00025;m.opt.impratio=10000.
    saved=np.load(args.design/f'candidate{args.candidate}.npz');qair=saved['q'].copy()
    initial=np.load(args.initial_state) if args.initial_state else None
    if initial is not None:
        if args.start!='held':raise ValueError('observed-state continuation requires held start')
        qair=initial['q'].copy()
        for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):qair[m.jnt_qposadr[j]+2]+=.15
    local=LocalGrasp(plant,qair,links)
    object_v=int(m.jnt_dofadr[m.joint('object').id]);object_q=int(m.jnt_qposadr[m.joint('object').id])
    # External support force is +(1-alpha)*m*g on the object. Since the
    # object is an independent rigid body, M*a below is exactly its negative.
    qground,supported,support_info,lift,carry,lower,upper=prepare(local,qair,plan)
    qdestination=qair.copy()
    for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
        qi=m.jnt_qposadr[j];qdestination[qi:qi+3]+=plan.destination[:3]-plan.source[:3]
    place=LocalTranslation(local,qdestination,np.array([0.,0.,-.15]),4.) if args.place else None
    N=lift.feedback['N'];n=N.shape[1];K=lift.feedback['K']
    (args.output/'design.json').write_text(json.dumps(dict(support=support_info,lift=lift.info,carry=carry.info,
                                                         preparation_s=time.perf_counter()-began),indent=2))
    d=mujoco.MjData(m);d.qpos[:]=qground;d.ctrl[:]=supported
    if initial is not None:d.qvel[:]=initial['v'];d.ctrl[:]=initial['ctrl']
    from amsrr_det.flight import FlightController
    flight=FlightController(plan.plant.pin,plan.plant.meta,plan.open_q)
    if args.flight=='coupled':
        import amsrr_det.coupled_flight as coupled
        from build.native_drive_interval_avx_fixed import NativeDriveInterval
        from build.native_bvls.solver import solve
        coupled.DrivePrediction=NativeDriveInterval;coupled.lsq_linear=solve
        flight=coupled.CoupledFlightController(plan.plant.pin,plan.plant.meta,plan.open_q,
                                               track_vectoring=True,planned_input_reference=True)
    qground_pin,_=plant.observation(d)
    acquisition=5. if args.start=='open' else 0.
    if args.geometric_acquisition:acquisition=20.
    if acquisition:d.qpos[:]=plant.configuration(plan.open_q)
    mujoco.mj_forward(m,d)
    audit=FastTaskAudit(plant,links,plan.shape.size,plan.support_top,shape=plan.shape,clearance_model=full.mj)
    delta=np.zeros(m.nv);records=[];times=[];clamps=0;reason=None;phase=None;began=time.perf_counter()
    grasp=None;acquired_dwell=0.;capture_planning_s=None
    from amsrr_det.grasp_control import GraspReferences
    target_geometry=GraspReferences(plan.plant.pin,plan.plant.meta,qground_pin,qground_pin,plan.surfaces,
                                   plan.points,plan.normals,samples=plan.samples,shape=plan.shape)
    desired_depth=target_geometry.geometry(qground_pin)[-1]
    for step in range(round(args.duration/.01)):
        before=time.perf_counter();t=float(d.time)-acquisition
        if args.geometric_acquisition and t>=0 and capture_planning_s is None:
            reason='acquisition_not_completed';break
        if t<0:
            current='ACQUIRE';r=smooth((t+acquisition)/4.)
            target=pin.interpolate(plan.plant.pin,plan.open_q,qground_pin,float(r))
            qo,vo=plant.observation(d)
            path_time=np.clip((t+acquisition)/4.,0.,1.)
            ds=(30*path_time**2-60*path_time**3+30*path_time**4)/4.
            tv=pin.difference(plan.plant.pin,plan.open_q,qground_pin)*ds
            if args.geometric_acquisition and d.time>=5.:
                if grasp is None:
                    grasp=GraspReferences(plan.plant.pin,plan.plant.meta,qo,qo,plan.surfaces,
                                          plan.points,plan.normals,samples=plan.samples,shape=plan.shape)
                oq=plan.plant.meta['object']['q']
                target,tv=grasp.update(qo,vo,qo[oq:oq+7],preload=desired_depth,normal_speed=.0002,object_feedback=0.)
                depth=grasp.geometry(qo)[-1]
                ready=np.all(depth>=.9*desired_depth) and grasp.relative_material_speed(qo,vo)<.001
                acquired_dwell=acquired_dwell+.01 if ready else 0.
                if acquired_dwell>=.3:
                    captured=time.perf_counter();qair=d.qpos.copy()
                    for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):qair[m.jnt_qposadr[j]+2]+=.15
                    local=LocalGrasp(plant,qair,links)
                    try:qground,supported,support_info,lift,carry,lower,upper=prepare(local,qair,plan)
                    except ValueError as error:reason='captured_design:'+str(error);break
                    K=lift.feedback['K'];N=lift.feedback['N'];acquisition=float(d.time)
                    capture_planning_s=time.perf_counter()-captured
                    (args.output/'captured_design.json').write_text(json.dumps(dict(depth_m=depth.tolist(),support=support_info,
                       lift=lift.info,carry=carry.info,planning_s=capture_planning_s),indent=2))
            joint=(1-r)*np.array([j['effort_nm'] for j in plan.trims[0]['joints']])+r*np.array([j['effort_nm'] for j in plan.trims[1]['joints']])
            force=(1-r)*np.array(plan.trims[0]['thrust'])+r*np.array(plan.trims[1]['thrust'])
            cmd,thrust=flight.update(qo,vo,target,target_velocity=tv,joint_bias=joint,thrust_reference=force)
            command=np.r_[np.where(local.dock,200.*cmd,cmd),thrust]
            reference=plant.configuration(target)
        elif t<3.:
            current='LOAD';r=smooth((t-1.)/2.)
            reference=qground;velocity=np.zeros(m.nv);u=(1-r)*supported+r*lift.center
        elif t<8.:
            current='LIFT';reference,velocity,u,_=lift.reference(t-3.+2.)
        elif not args.place or t<16.:
            current='CARRY';reference,velocity,u,_=carry.reference(t-8.+2.);K=carry.feedback['K'];N=carry.feedback['N']
        elif t<21.:
            current='PLACE';reference,velocity,u,_=place.reference(t-16.+2.);K=place.feedback['K'];N=place.feedback['N']
        else:
            current='UNLOAD';reference,velocity,_,_=place.reference(6.)
            r=smooth((t-21.)/2.);u=(1-r)*place.center+r*supported
        if t>=0:
            mujoco.mj_differentiatePos(m,delta,1.,reference,d.qpos)
            raw=u-K@np.r_[N.T@delta,N.T@(d.qvel-velocity)]
            if initial is not None and t<1.:
                blend=smooth(t);raw=(1-blend)*initial['ctrl']+blend*raw
            command=np.clip(raw,lower,upper);clamps+=int(np.any(abs(command-raw)>1e-10))
        d.ctrl[:]=command
        if current=='ACQUIRE':
            last_control=dict(q=qo.tolist(),v=vo.tolist(),target=target.tolist(),target_velocity=tv.tolist(),
                              command=command.tolist(),reference_diagnostics=grasp.last_diagnostics if grasp is not None else None,
                              flight_diagnostics=getattr(flight,'coupled_diagnostics',None))
        times.append(time.perf_counter()-before)
        if phase!=current:
            phase=current;print(json.dumps(dict(phase=phase,simulation_s=float(d.time),wall_s=time.perf_counter()-began)),flush=True)
            np.savez(args.output/f'entry_{phase}.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
        for _ in range(40):
            mujoco.mj_step(m,d);audit.update(d,carried=phase=='CARRY')
            reason=violation(audit)
            if reason:break
        if step%10==0:
            records.append(dict(time_s=float(d.time),phase=phase,object=d.qpos[object_q:object_q+7].tolist(),
                                reference=reference[object_q:object_q+7].tolist(),evaluator_loads=audit.last_normal_loads.tolist()))
        if reason:break
    stats=audit.result()
    # Combined MIT command is a torque encoding, not a position target.
    stats['maximum_servo_reference_error_rad']=None
    result=dict(scope=__doc__,start=args.start,initial_state=str(args.initial_state),flight=args.flight,geometric_acquisition=args.geometric_acquisition,
                desired_depth_m=desired_depth.tolist(),capture_planning_s=capture_planning_s,
                candidate=args.candidate,reason=reason or 'diagnostic_duration_reached',phase=phase,
                simulation_s=float(d.time),wall_s=time.perf_counter()-began,audit=stats,
                command_clamped_cycles=clamps,object_position_error_m=float(np.linalg.norm(d.qpos[object_q:object_q+3]-reference[object_q:object_q+3])),
                object_linear_speed_m_s=float(np.linalg.norm(d.qvel[object_v:object_v+3])),
                control_time_s=dict(mean=float(np.mean(times)),p95=float(np.quantile(times,.95)),maximum=max(times)))
    print(json.dumps(result),flush=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));(args.output/'trace.json').write_text(json.dumps(records))
    (args.output/'diagnostic.py').write_bytes(Path(__file__).read_bytes())
    if args.start=='open':(args.output/'last_acquisition_control.json').write_text(json.dumps(last_control,indent=2))
    np.savez(args.output/'final.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)


if __name__=='__main__':main()
