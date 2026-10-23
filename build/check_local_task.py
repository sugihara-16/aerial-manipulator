"""One continuous open-start grasp/transport evaluation of local mechanics.

Fresh generation uses static equations only; task simulation never selects a
candidate. --plan/--goal are explicit development replay options, not catalogues.
"""
import argparse,json,time,hashlib,sys
from functools import partial
from pathlib import Path
import numpy as np
import mujoco
import pinocchio as pin
import build.check_precise_contacts_100
from build.load_saved_plan import load_saved_plan
from build.mit_drive import dock_mit_model
from build.preload_endpoint import construct
from build.sticking_design import LocalGrasp
from build.local_translation import LocalTranslation,generated_translation,PredictiveTranslation
from build.touchdown_translation import TouchdownTranslation
from build.contact_step_control import ContactStepControl,pad_gap_jacobians
from build.material_travel import MaterialTravel
from build.check_local_pickup import violation,prepare,smooth
from build.algebraic_release import SupportedWithdrawal,coupled
from amsrr_det.mujoco_model import contact_dynamics_model
from amsrr_det.fast_task_audit import FastTaskAudit
from amsrr_det.numerics import quintic_reference
from amsrr_det.evidence import save_plan,save_sources
from build.background_plan import BackgroundPlan
from build.generated_grasp_stage import generate as generate_grasp_stage,entry_error

SOURCE_NAMES=['check_local_task.py','algebraic_release.py','unloaded_flight.py','unloaded_hover.py','mit_drive.py',
    'contact_step_control.py','sticking_design.py','local_translation.py','refine_captured_equilibrium.py',
    'preload_endpoint.py','linear_preload.py','held_input_solver.py','material_travel.py','check_local_pickup.py','support_contacts.py','contact_condensed_solver.py','local_contact_mpc.py','touchdown_translation.py','staged_approach.py','planned_withdrawal.py','hybrid_withdrawal.py','interval_tracking_control.py','response_contact_control.py','cached_response.py','finite_difference_response.py','short_contact_response.py','native_contact_rollout.py','contact_rollout.cc','contact_tangent.py','direct_response_step.py']
SOURCE_NAMES.extend(['full_state_feedback.py','background_plan.py','generated_grasp_stage.py','momentum_feedback.py','separation_rate_constraints.py','background_withdrawal.py','finite_slip_preload.py'])
SOURCE_NAMES.extend(['constrained_momentum_feedback.py','actuation_wrench_audit.py','background_withdrawal_hold.py'])
SOURCE_NAMES.extend(['finite_load_transfer.py','contact_unload.py'])
SOURCE_NAMES.extend(['contact_friction_modes.py','contact_allocation.py','contact_allocation_qp.py','contact_active_solver.py'])
SOURCE_NAMES.extend(['allocation_optimality.py','scaled_correction.py'])
SOURCE_NAMES.extend(['process_plan.py','supported_recovery.py'])
SOURCE_NAMES.append('planned_pad_loads.py')
SOURCE_NAMES.append('minimum_input_lp.py')
SOURCE_NAMES.extend(['select_structure.py','frozen_structure_trial.py'])
SOURCE_NAMES.extend(['task_wrench_margin.py','task_gimbal_design.py','translation_governor.py'])
LOADED_SOURCES={name:(Path('build')/name).read_bytes()for name in SOURCE_NAMES}

_ROOT=Path(__file__).resolve().parents[1]
_DEPENDENCIES={str(Path(module.__file__).resolve().relative_to(_ROOT)):Path(module.__file__).read_bytes()
    for module in list(sys.modules.values())if getattr(module,'__file__',None)
    and Path(module.__file__).resolve().is_relative_to(_ROOT/'build')and Path(module.__file__).is_file()}
_DEPENDENCIES[str(Path(__file__).resolve().relative_to(_ROOT))]=Path(__file__).read_bytes()
_DEPENDENCIES.update({'build/'+name:raw for name,raw in LOADED_SOURCES.items()})
for _name in ['build/prediction_native_avx_fixed/libdrive_interval.so','build/native_bvls/bvls.so',
              'build/body_bounds/python/mujoco/libmujoco.so.3.16.0']:
    _DEPENDENCIES[_name]=(_ROOT/_name).read_bytes()

# Capture the compiled stepping loop as well as its matching C++ source.
_contact_source=(_ROOT/'build/contact_rollout.cc').read_bytes()
_contact_digest=hashlib.sha256(_contact_source+mujoco.mj_versionString().encode()).hexdigest()[:16]
_contact_binary=_ROOT/('build/contact_rollout_'+_contact_digest+'.so')
if _contact_binary.exists():_DEPENDENCIES[str(_contact_binary.relative_to(_ROOT))]=_contact_binary.read_bytes()



def main(argv=None, observer=None):
    parser=argparse.ArgumentParser();src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--plan');src.add_argument('--request')
    parser.add_argument('--goal');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=110.)
    parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--fast-solve',action='store_true')
    parser.add_argument('--condensed-solve',action='store_true')
    parser.add_argument('--predictive-motion',action='store_true')
    parser.add_argument('--constrained-load',action='store_true')
    parser.add_argument('--interval-motion',action='store_true')
    parser.add_argument('--interval-acquisition',action='store_true')
    parser.add_argument('--generated-feedback',action='store_true')
    parser.add_argument('--rigid-object-feedback',action='store_true',help='diagnostic: generated outer feedback observes object rigid motion only')
    parser.add_argument('--momentum-feedback',action='store_true',help='comparison: generate outer feedback from whole momentum and physical input maps')
    parser.add_argument('--finite-slip-preload',action='store_true',help='comparison: match the nominal input to the finite friction law and retain acquisition control during refinement')
    parser.add_argument('--minimize-input',action='store_true',help='minimize the finite-contact reference thrust burden while retaining its existing reserve')
    parser.add_argument('--global-structure-search',action='store_true',help='compare 1-8 modules using physical bounds and fixed independent trial evidence')
    parser.add_argument('--structure-workers',type=int,default=8)
    parser.add_argument('--structure-generator',choices=['legacy','task','mixed'],default='task')
    parser.add_argument('--structure-seed',type=int,help='replay a recorded trial series; omitted means fresh independent series')
    parser.add_argument('--governed-motion',action='store_true',help='require task-direction authority and govern consistent path acceleration')
    parser.add_argument('--bounded-allocation',action='store_true')
    parser.add_argument('--hold-withdrawal-pose',action='store_true')
    parser.add_argument('--supported-load-transfer',action='store_true')
    parser.add_argument('--preload-floor',action='store_true',help='retain the generated preload reserve during acquisition refinement')
    parser.add_argument('--observed-handoff',action='store_true',help='refresh the finite holding reference at the settled acquisition observation')
    parser.add_argument('--contact-coupled',action='store_true')
    parser.add_argument('--nearby-handoff',action='store_true')
    parser.add_argument('--inertial-input-updates',action='store_true',help='regularize acquisition input jumps by their free-dynamics velocity effect')
    parser.add_argument('--planned-preload-floor',action='store_true',help='diagnostic: acquire the individual pad loads of the generated finite-slip reference')
    parser.add_argument('--finite-acquisition-support',action='store_true',help='use the finite support law in the existing acquisition solve')
    parser.add_argument('--process-withdrawal',action='store_true',help='isolate background withdrawal planning from the control interpreter')
    parser.add_argument('--normal-tracking',action='store_true',help='diagnostic: internal gap restoring objective in bounded momentum allocation')
    parser.add_argument('--internal-tracking',action='store_true',help='diagnostic: relative configuration restoring objective in bounded momentum allocation')
    parser.add_argument('--constrained-motion',action='store_true')
    parser.add_argument('--staged-approach',action='store_true')
    parser.add_argument('--planned-withdrawal',action='store_true')
    parser.add_argument('--hybrid-withdrawal',action='store_true')
    parser.add_argument('--background-withdrawal',action='store_true',help='continue unloaded control while preparing the ordinary withdrawal path')
    parser.add_argument('--distance-closing',action='store_true')
    parser.add_argument('--contact-counts',nargs='+',type=int,default=[2,3,4])
    args=parser.parse_args(argv)
    if args.steps<2 or args.steps%2:parser.error('physics steps must be a positive even number for 5 ms contact updates')
    out=args.output;out.mkdir(exist_ok=False);started=time.perf_counter()
    if args.rigid_object_feedback and not args.generated_feedback:raise ValueError('object-only comparison requires generated feedback')
    if args.momentum_feedback and (not args.generated_feedback or args.rigid_object_feedback):raise ValueError('momentum comparison requires generated feedback and excludes projected full-state gains')
    if args.finite_slip_preload and not args.momentum_feedback:raise ValueError('finite-slip preload requires momentum feedback')
    if args.minimize_input and not args.finite_slip_preload:raise ValueError('input minimization requires finite-slip preload')
    if args.global_structure_search and (not args.request or not args.minimize_input):raise ValueError('global structure search requires a fresh request and input minimization')
    if args.bounded_allocation and not args.momentum_feedback:raise ValueError('bounded allocation requires momentum feedback')
    if args.governed_motion and not(args.bounded_allocation and args.finite_slip_preload):raise ValueError('governed motion requires bounded finite-slip feedback')
    if args.hold_withdrawal_pose and not args.background_withdrawal:raise ValueError('held withdrawal pose requires background withdrawal')
    if args.supported_load_transfer and not args.finite_slip_preload:raise ValueError('supported transfer requires finite-slip preload')
    if args.preload_floor and not args.finite_slip_preload:raise ValueError('preload floor requires finite-slip preload')
    if args.observed_handoff and not args.finite_slip_preload:raise ValueError('observed handoff requires finite-slip preload')
    if args.planned_preload_floor and not args.preload_floor:raise ValueError('planned pad loads require preload floor')
    if args.process_withdrawal and not args.background_withdrawal:raise ValueError('process withdrawal requires background withdrawal')
    if args.normal_tracking and not args.bounded_allocation:raise ValueError('normal tracking requires bounded allocation')
    if args.internal_tracking and (not args.bounded_allocation or args.normal_tracking):raise ValueError('choose one internal objective with bounded allocation')
    if args.background_withdrawal and not args.hybrid_withdrawal:raise ValueError('background withdrawal requires hybrid withdrawal')
    if args.generated_feedback:
        if args.interval_motion or args.interval_acquisition:raise ValueError('choose one control architecture')
        args.constrained_motion=True
    if args.interval_motion:args.constrained_motion=True
    if args.constrained_motion:args.predictive_motion=True
    for name,source in LOADED_SOURCES.items():(out/name).write_bytes(source)
    hashes=save_sources(out)
    for name,raw in _DEPENDENCIES.items():
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    (out/'source_snapshot'/'manifest.json').write_text(json.dumps(hashes,indent=2))
    settings=dict(supported_recovery=True,aggregate_preload_domain=True,recovery_update_period_s=.005,contact_update_period_s=.0025,relative_acquisition=True,inertial_increment_model='one_period_contact',nominal_wrench_compensation=True,predictive_motion=args.predictive_motion,condensed_solve=args.condensed_solve,fast_solve=args.fast_solve,minimize_input=args.minimize_input,
                  global_structure_search=args.global_structure_search,structure_workers=args.structure_workers,structure_generator=args.structure_generator,governed_motion=args.governed_motion,
                  distance_closing=args.distance_closing,steps=args.steps,contact_counts=None if args.global_structure_search else args.contact_counts,
                  structure_contact_scope=('task contact roles; movable opposed pair; variable-count task auxiliary' if args.structure_generator=='task'
                                           else 'all nonempty free-pad subsets, 1..2N+2') if args.global_structure_search else 'legacy seed and branch search',
                  duration=args.duration,generated_feedback=args.generated_feedback,rigid_object_feedback=args.rigid_object_feedback,momentum_feedback=args.momentum_feedback,finite_slip_preload=args.finite_slip_preload,bounded_allocation=args.bounded_allocation,hold_withdrawal_pose=args.hold_withdrawal_pose,preload_floor=args.preload_floor,
                  contact_geometry_tolerance=1e-12 if args.generated_feedback else 1e-9,contact_geometry_iterations=1000 if args.generated_feedback else 100,
                  supported_load_transfer=args.supported_load_transfer,observed_handoff=args.observed_handoff,inertial_input_updates=args.inertial_input_updates,nearby_handoff=args.nearby_handoff,contact_coupled=args.contact_coupled,
                  constrained_load=args.constrained_load,interval_motion=args.interval_motion,interval_acquisition=args.interval_acquisition,constrained_motion=args.constrained_motion,staged_approach=args.staged_approach,planned_withdrawal=args.planned_withdrawal,hybrid_withdrawal=args.hybrid_withdrawal,background_withdrawal=args.background_withdrawal,
                  finite_acquisition_support=args.finite_acquisition_support,process_withdrawal=args.process_withdrawal,normal_tracking=args.normal_tracking,internal_tracking=args.internal_tracking,planned_preload_floor=args.planned_preload_floor)
    evidence=dict(scope='fresh_generation_spawn_task'if args.request else'saved_development_structure',method_scope=__doc__,
                  execution_entrypoint='fresh_task_input'if args.request else'saved_development_plan',engine='MuJoCo',
                  execution_configuration_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),
                  execution_configuration=settings,contact_engine=dict(sha256=hashes['build/body_bounds/python/mujoco/libmujoco.so.3.16.0']))

    (out/'experiment.json').write_text(json.dumps(evidence,indent=2))
    selected_prepared=None
    if args.global_structure_search:
        from build.select_structure import select_structure
        request=json.loads(Path(args.request).read_text())
        (out/'request.json').write_text(json.dumps(request,indent=2))
        try:
            plan,selected_prepared,selection=select_structure(request,workers=args.structure_workers,seed=args.structure_seed,task_wrench=args.governed_motion,generator=args.structure_generator)
        except ValueError as error:
            (out/'selection.json').write_text(json.dumps(getattr(error,'evidence',{}),separators=(',',':')))
            (out/'result.json').write_text(json.dumps(dict(**evidence,task_success=False,reason=str(error),initial_planning_s=time.perf_counter()-started)))
            return
        (out/'selection.json').write_text(json.dumps(selection,separators=(',',':')));candidates=[plan]
    elif args.plan:
        plan,request=load_saved_plan(args.plan);candidates=[plan]
    else:
        from amsrr_det.objects import ObjectShape
        from amsrr_det.structure_search import grasp_candidates
        request=json.loads(Path(args.request).read_text());payload=request['payload']
        shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
        candidates=grasp_candidates(shape,np.array(request['source']),np.array(request['destination']),request['support_top'],
            contact_counts=tuple(args.contact_counts),deadline=started+15.)
    (out/'request.json').write_text(json.dumps(request,indent=2));screens=[];goal=None
    candidates=iter(candidates)
    while True:
        try:plan=next(candidates)
        except StopIteration:break
        except ValueError as error:
            screens.append(dict(generation_error=str(error)));break
        links=[f'm{a}/'+('yaw_dock_mech'if b.startswith('yaw')else'pitch_dock_mech')+b[-1]for a,b in plan.surfaces]
        full=dock_mit_model(plan.plant,kp=200.);plant=contact_dynamics_model(full,links);m=plant.mj
        m.opt.impratio=10000.;m.opt.timestep=.01/args.steps
        if args.generated_feedback:
            # Small input-response derivatives need more geometric accuracy
            # than a single collision test; preserve every physical parameter.
            m.opt.ccd_tolerance=1e-12;m.opt.ccd_iterations=1000
        free=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        if selected_prepared is not None:
            goal=selected_prepared['qair'].copy();report=dict(reason='newly_selected_finite_reference',burden=selected_prepared['burden'])
        elif args.goal:
            goal=np.load(args.goal)['q'].copy();report=dict(reason='saved_development_endpoint')
        else:
            q=plant.configuration(plan.closed_q)
            for j in free:q[m.jnt_qposadr[j]+2]+=.15
            result,report=construct(plant,full.mj,q,links)
            if result is not None:goal=result[0]
        screens.append(dict(modules=plant.meta['modules'],surfaces=plan.surfaces,endpoint=report))
        if args.request and not args.global_structure_search and time.perf_counter()-started>15.:
            screens[-1]['deadline_exceeded']=True;goal=None;break
        if goal is not None:break
    initial_planning_s=time.perf_counter()-started
    (out/'screens.json').write_text(json.dumps(screens,indent=2))
    if goal is None:
        (out/'result.json').write_text(json.dumps(dict(**evidence,task_success=False,reason='no_static_endpoint',initial_planning_s=initial_planning_s)))
        return
    save_plan(plan,out/'generated_plan',__doc__);(out/'generated_plan'/'request.json').write_text(json.dumps(request))
    for j in free:goal[m.jnt_qposadr[j]+2]-=.15
    query=mujoco.MjData(m);query.qpos[:]=goal;mujoco.mj_fwdPosition(m,query)
    normal_targets=-pad_gap_jacobians(m,query,links)[0];qgoal,_=plant.observation(query)
    approach=None
    if args.staged_approach:
        from build.staged_approach import StagedApproach
        try:
            approach=(StagedApproach.from_prepared(plan,plant,links,selected_prepared['approach_path'])
                      if selected_prepared is not None else StagedApproach(plan,plant,goal,links))
        except ValueError as error:
            (out/'result.json').write_text(json.dumps(dict(**evidence,task_success=False,reason=str(error),phase='APPROACH_PLAN',initial_planning_s=time.perf_counter()-started)))
            return
        (out/'approach.json').write_text(json.dumps(approach.info,indent=2))
        initial_planning_s=time.perf_counter()-started
        if args.request and initial_planning_s>15.:
            (out/'result.json').write_text(json.dumps(dict(**evidence,task_success=False,reason='initial_planning_deadline',initial_planning_s=initial_planning_s)))
            return
    np.savez(out/'endpoint.npz',q=goal,normal_targets=normal_targets)
    flight=coupled.CoupledFlightController(plan.plant.pin,plan.plant.meta,plan.open_q,track_vectoring=True,planned_input_reference=True)
    control=ContactStepControl(plant,links,period=.0025,regularized_friction=True,finite_support=args.finite_acquisition_support,fast_solve=args.fast_solve,distance_closing=args.distance_closing,condensed_solve=args.condensed_solve,clearance_model=full.mj if args.momentum_feedback else None,inertial_updates=args.inertial_input_updates)
    d=mujoco.MjData(m);d.qpos[:]=plant.configuration(plan.open_q);mujoco.mj_forward(m,d)
    audit=FastTaskAudit(plant,links,plan.shape.size,plan.support_top,shape=plan.shape,clearance_model=full.mj)
    from build.actuation_wrench_audit import ActuationWrenchAudit
    actuation_audit=ActuationWrenchAudit(plant)if args.bounded_allocation else None
    material=MaterialTravel(plant,links);phase='APPROACH';ready=0.;entry=0.;reason=None;full_feedback=None
    grasp_job=None;generated_stage=None;preload_ready_time=0.;refinement_contact_time=0.;target_preload=0.
    handoff_job=None;handoff_checked=not args.observed_handoff;handoff_attempts=[]
    air_job=None;load_transfer=None;last_air_attempt=-np.inf;air_attempts=[];basis_observation_time=None
    trace=[];times=[];phase_times={};planning=[];prediction_errors=dict(samples=0,object_position_m=0.,joint_position_rad=0.,object_linear_velocity_m_s=0.,joint_velocity_rad_s=0.,pad_normal_force_n=0.);delta=np.zeros(m.nv);start=time.perf_counter();previous_phase=None
    oi=int(m.jnt_qposadr[m.joint('object').id]);ov=int(m.jnt_dofadr[m.joint('object').id])
    groups=control.groups;governors={};descent_probe_start=None
    recovery=None;recovery_events=[]
    for step in range(round(args.duration/.0025)):
        if d.time>=args.duration:break
        period=.005 if phase=='RECOVER' else (.0025 if phase in ['ACQUIRE','REFINE','LOAD','LIFT','CARRY','DESCEND'] else .01)
        before=time.perf_counter();prediction=None;query.qpos[:]=d.qpos;query.qvel[:]=d.qvel;mujoco.mj_fwdPosition(m,query)
        gaps,J=pad_gap_jacobians(m,query,links,groups)
        if phase not in ['UNLOAD','RETREAT']:material.update(d.qpos.copy())
        try:
            if phase=='APPROACH' and (min(gaps+.01*(J@d.qvel))<=0 or
                    (approach is not None and approach.ready(d.qpos,d.qvel,float(d.time)))):
                phase='ACQUIRE';anchor=d.ctrl.copy();period=.0025
            if phase=='APPROACH':
                target,tv,ta,r=quintic_reference(plan.plant.pin,plan.plant.meta,plan.open_q,qgoal,d.time,4.)
                q,v=plant.observation(d)
                effort=(1-r)*np.array([x['effort_nm']for x in plan.trims[0]['joints']])+r*np.array([x['effort_nm']for x in plan.trims[1]['joints']])
                thrust=(1-r)*np.array(plan.trims[0]['thrust'])+r*np.array(plan.trims[1]['thrust'])
                if approach is not None:target,tv,ta,r,effort,thrust=approach.reference(float(d.time))
                cmd,forces=flight.update(q,v,target,target_velocity=tv,target_acceleration=ta if approach is not None else None,joint_bias=effort,thrust_reference=thrust)
                command=np.r_[np.where(control.dock,200*cmd,cmd),forces]
            elif phase in ['ACQUIRE','REFINE']:
                design_entry_ready=max(gaps)<=0.
                if args.momentum_feedback and grasp_job is None and full_feedback is None:
                    from build.separation_rate_constraints import preload_in_physical_domain
                    from build.full_state_feedback import observed_tangent_speed
                    # Generating the stronger preload must not require that
                    # preload already exists. All pads must have made contact;
                    # the physical preload test remains at the LOAD handoff.
                    design_contact_ready=design_entry_ready if args.finite_slip_preload else preload_in_physical_domain(m,query,links)
                    preload_ready_time=preload_ready_time+period if (design_contact_ready and observed_tangent_speed(m,query,links)<1e-4)else 0.
                    design_entry_ready=design_entry_ready and preload_ready_time>=.3
                if args.generated_feedback and grasp_job is None and full_feedback is None and design_entry_ready:
                    np.savez(out/'grasp_design_observation.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
                    grasp_job=BackgroundPlan(d.time,partial(generate_grasp_stage,contact_coupled=args.contact_coupled,momentum_feedback=args.momentum_feedback,finite_slip_preload=args.finite_slip_preload,bounded_allocation=args.bounded_allocation,normal_tracking=args.normal_tracking,internal_tracking=args.internal_tracking,minimize_input=args.minimize_input,governed_motion=args.governed_motion,planned_reference=goal.copy()),plant,full.mj,d.qpos.copy(),links,plan)
                if grasp_job is not None and grasp_job.ready(d.time):
                    basis_observation_time=grasp_job.started_sim
                    generated_stage,wait=grasp_job.result(d.time);grasp_job=None
                    goal=generated_stage['goal'];normal_targets=generated_stage['normal_targets']
                    full_feedback=generated_stage['feedback'];qair=generated_stage['qair']
                    full_feedback.rigid_object_only=args.rigid_object_feedback
                    full_feedback.info['rigid_object_only']=args.rigid_object_feedback
                    if args.preload_floor:
                        finite=generated_stage['refinement']['finite_slip']
                        final=finite['iterations'][-1]['recheck'] if finite['iterations'] else finite['initial_fixed']
                        # The holding design retains 95% of its maximum shared
                        # reserve. Require that same physical preload in the
                        # acquisition solve, not only in its handoff predicate.
                        target_preload=.5*(1+.95*final['reserve']);refinement_contact_time=0.
                        if args.planned_preload_floor:
                            from build.planned_pad_loads import planned_pad_loads
                            target_preload=np.maximum(target_preload,planned_pad_loads(
                                LocalGrasp(plant,qair,links),qair,np.array(finite['nominal_velocity'])))
                        full_feedback.info['acquisition_minimum_normal_load_n']=np.asarray(target_preload).tolist()
                        full_feedback.info['acquisition_preload_activation']='all pads touching and existing tangent-speed criterion retained for 0.3 s'
                    (out/'refinement.json').write_text(json.dumps(generated_stage['refinement'],indent=2))
                    (out/'feedback_design.json').write_text(json.dumps(full_feedback.info,indent=2))
                    np.savez(out/'feedback_design.npz',q=qair,u=full_feedback.u,K=full_feedback.K)
                    phase='REFINE';ready=0.
                    planning.append(dict(phase=phase,background=True,**wait))
                if args.interval_acquisition and not hasattr(control,'predictor') and min(gaps)<=0.:
                    from build.interval_tracking_control import IntervalTrackingControl
                    control=IntervalTrackingControl(plant,links,clearance_model=full.mj,distance_closing=args.distance_closing)
                    control.previous=d.ctrl.copy()
                control.previous_input=d.ctrl.copy()
                if full_feedback is not None:
                    from build.full_state_feedback import observed_tangent_speed
                    if args.finite_slip_preload:
                        if args.preload_floor:
                            all_settled=max(gaps)<=0. and observed_tangent_speed(m,query,links)<1e-4
                            refinement_contact_time=refinement_contact_time+period if all_settled else 0.
                            # Do not demand an immediate positive force from a
                            # pad that lost contact while the basis was built.
                            control.minimum_normal_load=target_preload if refinement_contact_time>=.3 else 0.
                        from build.separation_rate_constraints import PreloadDomain
                        if not isinstance(control.separation,PreloadDomain):
                            control.separation=PreloadDomain(control.separation,m,links,period)
                        command=control.update(d.qpos.copy(),d.qvel.copy(),goal,input_reference=full_feedback.u,normal_targets=normal_targets)
                    else:command=full_feedback.update(d.qpos.copy(),d.qvel.copy(),goal)
                else:
                    from build.generated_grasp_stage import align_reference_to_object
                    acquisition_reference=align_reference_to_object(m,goal,d.qpos)
                    command=control.update(d.qpos.copy(),d.qvel.copy(),acquisition_reference,input_reference=anchor,normal_targets=normal_targets)
                # Readiness means the achieved compression remains in its
                # tolerance band under the next held input. An independent
                # absolute normal-velocity cutoff can prevent transition even
                # when every observed and predicted compression stays inside.
                next_gaps=gaps if full_feedback is not None else (control.predicted_gaps if hasattr(control,'predictor') else gaps+period*(J@np.asarray(control.diagnostics['predicted_velocity'])))
                compression_error=np.maximum(abs(gaps+normal_targets),abs(next_gaps+normal_targets))
                # Acquiring the desired compression permits a one-time
                # static refinement at the observed grasp. Settled contact is
                # still required AFTER that refinement, before carrying.
                settled=observed_tangent_speed(m,query,links)<1e-4 if full_feedback is not None else ((args.interval_acquisition and phase=='ACQUIRE') or control.diagnostics['observed_tangent_speed']<1e-4)
                preload_ready=np.all(compression_error<=.05*normal_targets)
                if args.momentum_feedback and full_feedback is not None:
                    from build.separation_rate_constraints import preload_in_physical_domain
                    preload_ready=preload_in_physical_domain(m,query,links)
                    if args.planned_preload_floor:
                        # The ordinary settled interval starts only once the
                        # acquisition solve is issuing the planned pad loads.
                        preload_ready=preload_ready and refinement_contact_time>=.3
                ready=ready+period if (preload_ready and settled)else 0.
                if ready>=.3 and full_feedback is not None and not handoff_checked and handoff_job is None:
                    # Refresh only the input/reference at the achieved geometry.
                    # Acquisition keeps running during the measured planning delay;
                    # the evaluator state and persistent material anchors stay intact.
                    operation=partial(generate_grasp_stage,contact_coupled=args.contact_coupled,momentum_feedback=True,finite_slip_preload=True,
                        bounded_allocation=args.bounded_allocation,minimize_input=args.minimize_input,
                        normal_tracking=args.normal_tracking,internal_tracking=args.internal_tracking,
                        governed_motion=args.governed_motion,fixed_observation=True,nearby_observation=args.nearby_handoff,
                        material_travel_used=float(material.peak.max(initial=0.)))
                    handoff_job=BackgroundPlan(d.time,operation,plant,full.mj,d.qpos.copy(),links,plan)
                if handoff_job is not None and handoff_job.ready(d.time):
                    candidate,wait=handoff_job.result(d.time);handoff_job=None
                    handoff_checked=True
                    planning.append(dict(phase='OBSERVED_HANDOFF',background=True,**wait))
                    handoff_attempts.append(dict(time_s=float(d.time),available=candidate['available'],
                        retained_previous_reference=not candidate['available'],
                        refinement=candidate['refinement'],**wait))
                    (out/'observed_handoff.json').write_text(json.dumps(handoff_attempts,indent=2))
                    if candidate['available']:
                        generated_stage=candidate;full_feedback=candidate['feedback'];qair=candidate['qair']
                        goal=candidate['goal'];normal_targets=candidate['normal_targets']
                        if args.nearby_handoff:ready=0.
                        (out/'refinement.json').write_text(json.dumps(candidate['refinement'],indent=2))
                        (out/'feedback_design.json').write_text(json.dumps(full_feedback.info,indent=2))
                        np.savez(out/'feedback_design.npz',q=qair,u=full_feedback.u,K=full_feedback.K)
                    # If the fixed-pose approximation is unresolved, retain the
                    # already constructed reference. This optional update is
                    # not a new necessary condition for a physical transition.
                if ready>=.3 and handoff_checked and (not args.generated_feedback or full_feedback is not None) and not args.supported_load_transfer and (not args.governed_motion or full_feedback is None or entry_error(full_feedback,d.qpos,d.qvel,goal)<=1.):
                    began=time.perf_counter();ready=0.;qair=d.qpos.copy()
                    for j in free:qair[m.jnt_qposadr[j]+2]+=.15
                    if phase=='ACQUIRE':
                        endpoint,info=construct(plant,full.mj,qair,links)
                        (out/'refinement.json').write_text(json.dumps(info,indent=2))
                        if endpoint is None:raise ValueError('acquired_static_endpoint_not_found')
                        goal=endpoint[0].copy()
                        for j in free:goal[m.jnt_qposadr[j]+2]-=.15
                        query.qpos[:]=goal;mujoco.mj_fwdPosition(m,query)
                        normal_targets=-pad_gap_jacobians(m,query,links,groups)[0]
                        phase='REFINE';anchor=command.copy()
                        if hasattr(control,'reset_search'):control.reset_search()
                    else:
                        # Synthesize around the computed feasible nominal,
                        # and feed the actual observed error to feedback.
                        # A nearby observation need not be a static equilibrium.
                        qair=goal.copy()
                        for j in free:qair[m.jnt_qposadr[j]+2]+=.15
                        if generated_stage is not None:
                            ground,supported,support_info,lift,carry,lower,upper=generated_stage['transfer']
                        else:
                            local=LocalGrasp(plant,qair,links)
                            displacement=plan.destination[:3]-d.qpos[oi:oi+3]
                            ground,supported,support_info,lift,carry,lower,upper=prepare(local,qair,plan,displacement,predictive=args.predictive_motion,
                                motion_factory=partial(PredictiveTranslation,reference_only=True) if args.interval_motion else
                                (PredictiveTranslation if args.constrained_motion else None))
                        if args.interval_motion:
                            from build.interval_tracking_control import IntervalTrackingControl
                            interval_control=IntervalTrackingControl(plant,links,clearance_model=full.mj)
                            interval_control.previous=d.ctrl.copy()
                            interval_phase='LOAD'
                        elif args.constrained_load:
                            from build.support_contacts import support_contacts
                            from build.local_contact_mpc import LocalContactMPC
                            grounded=LocalGrasp(plant,ground,links,support=support_contacts(m,ground))
                            load_control=LocalContactMPC(grounded,ground,fast_solve=True)
                        phase='LOAD';entry=float(d.time);handoff=command.copy()if args.finite_slip_preload else d.ctrl.copy()
                    planning.append(dict(phase=phase,wall_s=time.perf_counter()-began))
                if args.supported_load_transfer and phase=='REFINE':
                    from build.finite_load_transfer import FiniteLoadTransfer
                    load_transfer=FiniteLoadTransfer(plant,full.mj,links,goal,full_feedback.u,normal_targets)
                    np.savez(out/'supported_seed_design.npz',q=qair,u=full_feedback.u)
                    (out/'supported_seed_refinement.json').write_text(json.dumps(generated_stage['refinement'],indent=2))
                    phase='SUPPORT_TRANSFER'
            elif phase=='SUPPORT_TRANSFER':
                command=load_transfer.update(d.qpos.copy(),d.qvel.copy(),d.time)
                if air_job is None and load_transfer.ready and d.time-last_air_attempt>=.1:
                    last_air_attempt=float(d.time)
                    operation=partial(generate_grasp_stage,contact_coupled=args.contact_coupled,momentum_feedback=True,finite_slip_preload=True,
                        bounded_allocation=args.bounded_allocation,normal_tracking=args.normal_tracking,internal_tracking=args.internal_tracking,airborne_observation=True,minimize_input=args.minimize_input,
                        material_travel_used=float(material.peak.max(initial=0.)),governed_motion=args.governed_motion)
                    air_job=BackgroundPlan(d.time,operation,plant,full.mj,d.qpos.copy(),links,plan)
                if air_job is not None and air_job.ready(d.time):
                    observed_at=air_job.started_sim
                    candidate,wait=air_job.result(d.time);air_job=None
                    air_attempts.append(dict(observation_s=observed_at,available=candidate['available'],
                                             refinement=candidate['refinement'],**wait))
                    if candidate['available'] and load_transfer.ready:
                        generated_stage=candidate;goal=candidate['goal'];qair=candidate['qair']
                        normal_targets=candidate['normal_targets'];full_feedback=candidate['feedback']
                        ground,supported,support_info,lift,carry,lower,upper=candidate['transfer']
                        basis_observation_time=observed_at
                        (out/'refinement.json').write_text(json.dumps(candidate['refinement'],indent=2))
                        (out/'feedback_design.json').write_text(json.dumps(full_feedback.info,indent=2))
                        np.savez(out/'feedback_design.npz',q=qair,u=full_feedback.u,K=full_feedback.K)
                        planning.append(dict(phase='AIR_HANDOFF',background=True,observation_s=observed_at,**wait))
                        phase='LOAD';entry=float(d.time);handoff=command.copy()
            elif phase=='RECOVER':
                command=recovery.update(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time),float(material.peak.max(initial=0.)))
                if recovery.stage is not None:
                    generated_stage=recovery.stage;full_feedback=generated_stage['feedback']
                    goal=generated_stage['goal'];qair=generated_stage['qair'];normal_targets=generated_stage['normal_targets']
                    ground,supported,support_info,lift,carry,lower,upper=generated_stage['transfer']
                    basis_observation_time=recovery.info['observation_s']
                    recovery_events[-1].update(completed_s=float(d.time),details=recovery.info)
                    recovery.close();recovery=None
                    phase='LOAD';entry=float(d.time);handoff=command.copy();governors={}
            elif phase in ['LOAD','LIFT','CARRY']:
                t=float(d.time)-entry
                if t<3.:
                    phase='LOAD';r=smooth((t-1.)/2.);reference=ground;velocity=np.zeros(m.nv);u=(1-r)*supported+r*lift.center
                elif t<8.:
                    phase='LIFT';reference,velocity,u,_=lift.reference(t-3.+2.)
                else:
                    phase='CARRY';reference,velocity,u,_=carry.reference(t-8.+2.)
                if args.generated_feedback:
                    nominal=None
                    if args.finite_slip_preload and t<3.:
                        ratio=smooth(t/3.)
                        nominal=(1-ratio)*handoff+ratio*full_feedback.u
                    options={}
                    if args.governed_motion and phase in ['LIFT','CARRY']:
                        from build.translation_governor import TranslationGovernor
                        if phase not in governors:
                            if phase=='CARRY' and not governors['LIFT'].finished:raise ValueError('governed lift exceeded stage budget')
                            motion=lift if phase=='LIFT'else carry
                            governors[phase]=TranslationGovernor(m,motion.q,motion.delta,motion.duration,period=.0025)
                        if governors[phase].finished:
                            reference,velocity=governors[phase].reference()
                        else:options['path']=governors[phase]
                    try:
                        command=full_feedback.update(d.qpos.copy(),d.qvel.copy(),reference,velocity,input_reference=nominal,**options)
                    except ValueError as error:
                        if not args.contact_coupled or phase!='LOAD' or str(error) not in ['constant constraint infeasible','contact allocation nonlinear force check unresolved','contact allocation PrimalInfeasible']:
                            raise
                        from build.supported_recovery import SupportedRecovery
                        recovery=SupportedRecovery(plant,full.mj,plan,links,goal,normal_targets)
                        command=recovery.update(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time),float(material.peak.max(initial=0.)))
                        recovery_events.append(dict(started_s=float(d.time),reason=str(error),entry=recovery.info.copy()))
                        phase='RECOVER'
                elif not args.interval_motion:
                    feedback=lift.feedback if t<8. else carry.feedback
                    N,K=feedback['N'],feedback['K']
                    mujoco.mj_differentiatePos(m,delta,1.,reference,d.qpos)
                    command=u-K@np.r_[N.T@delta,N.T@(d.qvel-velocity)]
                if args.interval_motion:
                    if interval_phase!=phase:
                        interval_control.reset_search();interval_phase=phase
                    preferred=(1-smooth(t))*handoff+smooth(t)*u if t<1. else u
                    interval_control.minimum_normal_load=.5 if phase=='CARRY' else 0.
                    command=interval_control.update(d.qpos.copy(),d.qvel.copy(),reference,velocity,input_reference=preferred,normal_targets=normal_targets)
                    prediction=interval_control
                elif args.constrained_load and t<3.:
                    preferred=(1-smooth(t))*handoff+smooth(t)*u if t<1. else u
                    command=load_control.update(d.qpos.copy(),d.qvel.copy(),[(ground,velocity)]*load_control.H,input_reference=preferred)
                    prediction=load_control
                elif args.predictive_motion and t>=3. and not args.generated_feedback:
                    motion,motion_time=(lift,t-3.+2.)if t<8.else(carry,t-8.+2.)
                    command=motion.update(d.qpos.copy(),d.qvel.copy(),motion_time);prediction=getattr(motion,'mpc',None)
                if t<1. and not(args.constrained_load or args.interval_motion or args.generated_feedback):r=smooth(t);command=(1-r)*handoff+r*command
                if t>=16.:
                    if args.governed_motion and not governors['CARRY'].finished:raise ValueError('governed carry exceeded stage budget')
                    began=time.perf_counter()
                    if generated_stage is not None:descent=generated_stage['descent']
                    else:
                        nominal=carry.reference(100.)[0]if args.predictive_motion else d.qpos.copy()
                        query.qpos[:]=nominal;qpin,_=plant.observation(query);o=plant.meta['object']['q']
                        drop=plan.shape.lowest_z(qpin[o:o+7])-plan.support_top
                        motion=TouchdownTranslation if args.constrained_motion else (generated_translation if args.predictive_motion else LocalTranslation)
                        kwargs={'reference_only':True} if args.interval_motion else {}
                        descent=motion(LocalGrasp(plant,nominal,links),nominal,[0.,0.,-drop],4.,**kwargs)
                    planning.append(dict(phase='DESCEND',wall_s=time.perf_counter()-began))
                    if not(args.interval_motion or args.generated_feedback):N=descent.feedback['N'];K=descent.feedback['K']
                    phase='DESCEND';descent_entry=float(d.time);handoff=d.ctrl.copy()
            elif phase=='DESCEND':
                if args.governed_motion and float(d.time)-entry>23.:raise ValueError('governed motion exceeded finite-reference horizon')
                obj=m.geom('object').id
                supported_now=any(obj in c.geom and any(m.geom(int(g)).name in ['support','destination_support','floor']for g in c.geom)and c.dist<=0 for c in query.contact)
                if supported_now:
                    if args.contact_coupled and args.hold_withdrawal_pose:
                        from build.contact_unload import ContactUnload
                        release=ContactUnload(plan,plant,d.qpos.copy(),parallel_process=args.process_withdrawal)
                    elif args.hybrid_withdrawal:
                        from build.hybrid_withdrawal import HybridWithdrawal
                        if args.background_withdrawal:
                            from build.background_withdrawal import BackgroundHybridWithdrawal
                            release_type=BackgroundHybridWithdrawal
                            if args.hold_withdrawal_pose:
                                from build.background_withdrawal_hold import HeldBackgroundWithdrawal
                                release_type=HeldBackgroundWithdrawal
                            release=release_type(plan,plant,d.qpos.copy(),parallel_process=args.process_withdrawal)
                        else:release=HybridWithdrawal(plan,plant,d.qpos.copy())
                    elif args.planned_withdrawal:
                        from build.planned_withdrawal import PlannedWithdrawal
                        release=PlannedWithdrawal(plan,plant,d.qpos.copy())
                    else:release=SupportedWithdrawal(plan,plant,d.qpos.copy())
                    phase='UNLOAD'
                    command=release.update(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy())
                else:
                    t=float(d.time)-descent_entry;reference,velocity,u,_=descent.reference(t)
                    if args.generated_feedback:
                        options={}
                        if args.governed_motion:
                            from build.translation_governor import TranslationGovernor
                            if phase not in governors:governors[phase]=TranslationGovernor(m,descent.q,descent.delta,descent.duration,period=.0025)
                            path=governors[phase]
                            if not path.finished:options['path']=path
                            else:
                                if descent_probe_start is None:descent_probe_start=float(d.time)
                                reference,velocity,_,_=descent.reference(2.+descent.duration+float(d.time)-descent_probe_start)
                        command=full_feedback.update(d.qpos.copy(),d.qvel.copy(),reference,velocity,**options)
                    elif not args.interval_motion:
                        mujoco.mj_differentiatePos(m,delta,1.,reference,d.qpos)
                        command=u-K@np.r_[N.T@delta,N.T@(d.qvel-velocity)]
                    if args.interval_motion:
                        if interval_phase!=phase:
                            interval_control.reset_search();interval_phase=phase
                        interval_control.minimum_normal_load=0.
                        command=interval_control.update(d.qpos.copy(),d.qvel.copy(),reference,velocity,input_reference=u,normal_targets=normal_targets)
                        prediction=interval_control
                    elif args.predictive_motion and not args.generated_feedback:
                        command=descent.update(d.qpos.copy(),d.qvel.copy(),t)
                        if t>=1.:prediction=getattr(descent,'mpc',None)
                    if t<1. and not args.constrained_motion:r=smooth(t);command=(1-r)*handoff+r*command
            else:
                command=release.update(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy());phase='RETREAT'if release.ready else'UNLOAD'
                if release.complete:
                    qp,vp=plant.observation(d);o=plant.meta['object'];pose=qp[o['q']:o['q']+7];vel=vp[o['v']:o['v']+6]
                    angle=np.linalg.norm(pin.log3(pin.Quaternion(pose[3:]).matrix()@pin.Quaternion(plan.destination[3:]).matrix().T))
                    if (np.linalg.norm(pose[:3]-plan.destination[:3])<.03 and angle<.2 and
                        np.linalg.norm(vel[:3])<.02 and np.linalg.norm(vel[3:])<.1 and min(gaps)>.02 and
                        audit.last_contact_time is not None and d.time-audit.last_contact_time>=2.):
                        reason='task_completed';break
        except (ValueError,np.linalg.LinAlgError,mujoco.FatalError)as error:reason=str(error);break
        finally:
            elapsed=time.perf_counter()-before;times.append(elapsed);phase_times.setdefault(phase,[]).append(elapsed)
        if phase!=previous_phase:
            previous_phase=phase;print(json.dumps(dict(phase=phase,t=float(d.time))),flush=True)
            np.savez(out/f'entry_{phase}.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl,actuator_biasprm=m.actuator_biasprm)
        d.ctrl[:]=command
        if actuation_audit is not None:
            if phase in ['LOAD','LIFT','CARRY','DESCEND'] and hasattr(full_feedback,'requested_world_wrench'):
                actuation_audit.issue(d.qpos,command,full_feedback.requested_world_wrench)
            else:actuation_audit.request=None
        for _ in range(round(period/m.opt.timestep)):
            try:
                mujoco.mj_step(m,d);audit.update(d,carried=phase=='CARRY');reason=violation(audit)
                if actuation_audit is not None:actuation_audit.after_step(d)
            except mujoco.FatalError as error:
                reason='physics_or_geometry_error:'+str(error);break
            # The documented actuator model clips combined torque at its
            # continuous rating. Saturation is recorded, not a new task
            # rejection criterion; local unsaturated proofs do not cover it.
            if audit.max_effort_ratio>1.000001:reason='motor_effort_limit'
            if reason:break
        if prediction is not None:
            mujoco.mj_differentiatePos(m,delta,1.,prediction.predicted_q,d.qpos)
            velocity_error=d.qvel-prediction.predicted_v;prediction_errors['samples']+=1
            for key,value in dict(object_position_m=np.linalg.norm(delta[ov:ov+3]),joint_position_rad=max(abs(delta[control.vi])),
                object_linear_velocity_m_s=np.linalg.norm(velocity_error[ov:ov+3]),joint_velocity_rad_s=max(abs(velocity_error[control.vi])),
                pad_normal_force_n=max(abs(audit.last_normal_loads-prediction.predicted_loads))).items():
                prediction_errors[key]=max(prediction_errors[key],float(value))
        if step%200==0:
            trace.append(dict(time_s=float(d.time),phase=phase,gaps=gaps.tolist(),loads=audit.last_normal_loads.tolist()))
            print(json.dumps(trace[-1]),flush=True)
        # Optional display hook: the observer reads state, never advances physics.
        if observer is not None and not observer(m,d,phase):
            reason='viewer_interrupted'
        if reason:break
    if recovery is not None:recovery.close()
    if grasp_job is not None:grasp_job.close()
    if handoff_job is not None:handoff_job.close()
    if air_job is not None:air_job.close()
    if 'release' in locals() and hasattr(release,'close'):release.close()
    stats=audit.result();stats['maximum_servo_reference_error_rad']=None
    motion_methods={name:motion.info['method']for name,motion in [('lift',locals().get('lift')),('carry',locals().get('carry')),('descent',locals().get('descent'))]if motion is not None}
    qp,vp=plant.observation(d);o=plant.meta['object'];pose=qp[o['q']:o['q']+7]
    result=dict(**evidence,predictive_motion=args.predictive_motion,task_success=reason=='task_completed',reason=reason or 'task_timeout',phase=phase,fast_solve=args.fast_solve,distance_closing=args.distance_closing,condensed_solve=args.condensed_solve,
        object_position_error_m=float(np.linalg.norm(pose[:3]-plan.destination[:3])),
        object_orientation_error_rad=float(np.linalg.norm(pin.log3(pin.Quaternion(pose[3:]).matrix()@pin.Quaternion(plan.destination[3:]).matrix().T))),
        object_velocity=vp[o['v']:o['v']+6].tolist(),
        initial_planning_s=initial_planning_s,planning=planning,simulation_s=float(d.time),wall_s=time.perf_counter()-start,
        governed_paths={name:path.info for name,path in governors.items()},
        audit=stats,material=material.result(),control_time_s=dict(mean=float(np.mean(times)),p95=float(np.quantile(times,.95)),maximum=max(times)),
        prediction_errors=prediction_errors,motion_methods=motion_methods,phase_control_time_s={p:dict(mean=float(np.mean(t)),p95=float(np.quantile(t,.95)),maximum=max(t))for p,t in phase_times.items()},
        contacts=len(links),modules=plant.meta['modules'],release=release.info if phase in ['UNLOAD','RETREAT']else None,
        control_timing_includes_failed_calls=True)
    result['supported_recovery']=recovery_events
    result['finite_basis_observation_s']=basis_observation_time
    if load_transfer is not None:result['support_load_transfer']=dict(final=load_transfer.info,airborne_attempts=air_attempts)
    if args.interval_acquisition and hasattr(control,'predictor'):
        result['acquisition_derivative_cache']=control.predictor.counts
        control.predictor.close()
    if actuation_audit is not None:result['actuation_wrench']=actuation_audit.result()
    if full_feedback is not None:
        result['generated_feedback']=dict(**full_feedback.info,clipped_control_periods=full_feedback.clipped_periods,
            timing_scope=('grasp feedback and transfer planning run while control continues; ordinary withdrawal planning also runs in background; legacy rotor transition remains synchronous'
                          if args.background_withdrawal else 'grasp feedback and transfer planning run while control continues; withdrawal planning remains synchronous'))
    if args.interval_motion and 'interval_control' in locals():
        result['interval_derivative_cache']=interval_control.predictor.counts
        result['motion_command_method']='nonlinear_observed_state_interval'
        interval_control.predictor.close()
    if args.condensed_solve:
        from build.contact_condensed_solver import STATISTICS
        result['condensed_solver']=STATISTICS
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'trace.json').write_text(json.dumps(trace,indent=2))
    np.savez(out/'final.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
