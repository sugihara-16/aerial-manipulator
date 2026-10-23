"""Static grasp and its local feedback/path preparation from one observation."""
from functools import partial
import numpy as np
import mujoco
from build.preload_endpoint import construct
from build.sticking_design import LocalGrasp
from build.full_state_feedback import FullStateFeedback
from build.contact_step_control import pad_gap_jacobians
from build.check_local_pickup import prepare
from build.local_translation import PredictiveTranslation
from build.touchdown_translation import TouchdownTranslation


def entry_error(feedback,q,v,reference):
    from build.momentum_feedback import TRACKING_TOLERANCES
    oi=feedback.oi;angle=np.empty(3);rotation=np.empty(9)
    mujoco.mju_subQuat(angle,q[oi+3:oi+7],reference[oi+3:oi+7]);mujoco.mju_quat2Mat(rotation,reference[oi+3:oi+7])
    error=np.r_[q[oi:oi+3]-reference[oi:oi+3],rotation.reshape(3,3)@angle,np.linalg.solve(feedback.inertia,feedback.momentum(q,v))]
    return float(np.sum(abs(error)/TRACKING_TOLERANCES))


def align_reference_to_object(model,reference,observation):
    """Reuse this task's planned geometry at the currently observed object pose."""
    oi=int(model.jnt_qposadr[model.joint('object').id])
    before=np.empty((3,3));after=np.empty((3,3))
    mujoco.mju_quat2Mat(before.ravel(),reference[oi+3:oi+7])
    mujoco.mju_quat2Mat(after.ravel(),observation[oi+3:oi+7])
    turn=after@before.T;answer=reference.copy()
    for joint in np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
        index=int(model.jnt_qposadr[joint]);rotation=np.empty((3,3))
        mujoco.mju_quat2Mat(rotation.ravel(),reference[index+3:index+7])
        answer[index:index+3]=observation[oi:oi+3]+turn@(reference[index:index+3]-reference[oi:oi+3])
        mujoco.mju_mat2Quat(answer[index+3:index+7],(turn@rotation).ravel())
    return answer


def generate(plant,clearance_model,observed_q,links,plan,*,momentum_feedback=False,finite_slip_preload=False,bounded_allocation=False,
             airborne_observation=False,material_travel_used=0.,normal_tracking=False,internal_tracking=False,minimize_input=False,governed_motion=False,planned_reference=None,fixed_observation=False,nearby_observation=False,contact_coupled=False):
    m=plant.mj
    if fixed_observation and not finite_slip_preload:
        raise ValueError('fixed observation requires the finite contact law')
    z=m.jnt_qposadr[np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)]+2
    lift_height=.15
    if airborne_observation:
        if not finite_slip_preload:raise ValueError('airborne handoff requires the finite contact basis')
        query=mujoco.MjData(m);query.qpos[:]=observed_q;qp,_=plant.observation(query);oq=plant.meta['object']['q']
        lift_height=plan.support_top+.15-plan.shape.lowest_z(qp[oq:oq+7])
    qair=observed_q.copy();qair[z]+=lift_height
    finite_input=None;finite_info=None;observed_finite=None
    if finite_slip_preload:
        from build.finite_slip_preload import finite_slip_preload
        observed_finite,observed_check=finite_slip_preload(LocalGrasp(plant,qair,links),qair,23.,move_pose=False,
            travel_budget=.03-material_travel_used,minimize_input=minimize_input)
        if observed_finite is not None and not observed_check['feasible']:observed_finite=None
    if nearby_observation and observed_finite is None:
        from build.finite_slip_preload import construct_finite_slip
        try:
            observed_finite,nearby_check=construct_finite_slip(plant,clearance_model,qair,links,23.,minimize_input=minimize_input,travel_budget=.03-material_travel_used)
            observed_check=nearby_check
            qair=observed_finite[0].copy()
        except ValueError as error:
            return dict(available=False,refinement=dict(reason=str(error)))
    if (airborne_observation or fixed_observation) and observed_finite is None:
        return dict(available=False,refinement=observed_check)
    reference_info=None
    if observed_finite is None and finite_slip_preload and planned_reference is not None:
        # A changed contact active set can defeat local linearization at the
        # observation. Reuse only this task's geometric seed, then recheck the
        # finite law and all physical constraints; execution still moves there.
        from build.finite_slip_preload import construct_finite_slip
        seed=align_reference_to_object(m,planned_reference,qair)
        try:
            observed_finite,reference_info=construct_finite_slip(plant,clearance_model,seed,links,23.,minimize_input=minimize_input)
        except ValueError:
            observed_finite=None
    if observed_finite is None:
        endpoint,info=construct(plant,clearance_model,qair,links)
        if endpoint is None:
            if not finite_slip_preload:
                raise ValueError('acquired_static_endpoint_not_found:'+info['reason'])
            # Ideal sticking is an initializer, not a necessary condition
            # for the finite contact law used by this controller. Retain the
            # observation as a geometric seed and require the full finite
            # pose/force/command checks below; no static success is invented.
            info=dict(method='observed_geometry_seed',static_initialization=info)
        else:qair=endpoint[0].copy()
    else:info=(dict(method='task_reference_warm_start',observed_fixed=observed_check)
               if reference_info is not None else dict(method='observed_pose_finite_law',initial_fixed=observed_check))
    if finite_slip_preload:
        if not momentum_feedback:raise ValueError('finite-slip comparison requires momentum feedback')
        from build.finite_slip_preload import construct_finite_slip
        # LOAD 3 + LIFT 5 + CARRY 8 + DESCEND 6 + ordinary unload .3 s.
        if observed_finite is None:
            (qair,finite_input,finite_velocity),finite_info=construct_finite_slip(plant,clearance_model,qair,links,23.,minimize_input=minimize_input)
        else:
            from amsrr_det.clearance import MotionBoundedClearance
            if MotionBoundedClearance(clearance_model,links).check(qair,True)[0]:raise ValueError('observed finite pose clearance')
            qair,finite_input,finite_velocity=observed_finite
            finite_info=reference_info if reference_info is not None else (observed_check if nearby_observation and "iterations" in observed_check else dict(initial_fixed=observed_check,iterations=[]))
        if governed_motion and fixed_observation and not nearby_observation:
            from build.task_wrench_margin import task_wrench_margin
            margin=task_wrench_margin(LocalGrasp(plant,qair,links),qair,finite_input,
                                     plan.destination[:3]-plan.source[:3])
            if not margin['nominal_feasible'] or margin['normalized_feedback_l1_radius'] is None or margin['normalized_feedback_l1_radius']<1-1e-7:
                return dict(available=False,refinement=dict(finite_slip=finite_info,task_wrench=margin))
            info['task_wrench']=dict(margin=margin,repairs=[])
        elif governed_motion:
            from build.task_wrench_margin import refine_task_reference
            try:
                (qair,finite_input,finite_velocity),finite_info,wrench_info=refine_task_reference(
                    plant,clearance_model,qair,finite_input,finite_velocity,links,
                    plan.destination[:3]-plan.source[:3],finite_info)
            except ValueError as error:
                if not fixed_observation:raise
                # This optional replacement must pass the full design check.
                # Failure leaves the already verified reference with its caller.
                return dict(available=False,refinement=dict(reason=str(error),evidence=getattr(error,'evidence',{})))
            info['task_wrench']=wrench_info
        finite_info['nominal_velocity']=finite_velocity.tolist()
        info['finite_slip']=finite_info
        if fixed_observation:info['method']='nearby_observation_finite_law' if nearby_observation else 'settled_observation_finite_law'
    goal=qair.copy();goal[z]-=lift_height
    local=LocalGrasp(plant,qair,links)
    if momentum_feedback:
        from build.momentum_feedback import MomentumFeedback
        feedback_type=MomentumFeedback
        if bounded_allocation:
            from build.constrained_momentum_feedback import ConstrainedMomentumFeedback
            feedback_type=ConstrainedMomentumFeedback
        options=dict(normal_tracking=normal_tracking,internal_tracking=internal_tracking,contact_coupled=contact_coupled) if bounded_allocation else {}
        feedback=feedback_type(local,qair,finite_input,period=.01,**options)
    else:feedback=FullStateFeedback(local,qair,normal_margin_cost=True)
    query=mujoco.MjData(m);query.qpos[:]=goal;mujoco.mj_fwdPosition(m,query)
    targets=-pad_gap_jacobians(m,query,links)[0]
    oi=int(m.jnt_qposadr[m.joint('object').id])
    # A displacement applies to this reference, so use its own starting pose.
    displacement=plan.destination[:3]-goal[oi:oi+3]
    if finite_slip_preload:
        # The momentum controller uses the generated finite-contact input.
        # These objects supply only geometric position/velocity references;
        # an unused ideal-sticking force solve is not a validity condition.
        lift=PredictiveTranslation(local,goal,[0.,0.,lift_height],4.,reference_only=True,input_reference=finite_input)
        carry=PredictiveTranslation(local,qair,displacement,6.,reference_only=True,input_reference=finite_input)
        transfer=(goal,finite_input,dict(scope='air input; support allocation is not used by this controller'),lift,carry,None,None)
    else:
        transfer=prepare(local,qair,plan,displacement,predictive=True,
                         motion_factory=partial(PredictiveTranslation,reference_only=True))
    carry=transfer[4];nominal=carry.reference(100.)[0]
    query.qpos[:]=nominal;qpin,_=plant.observation(query);oq=plant.meta['object']['q']
    drop=plan.shape.lowest_z(qpin[oq:oq+7])-plan.support_top
    descent=TouchdownTranslation(LocalGrasp(plant,nominal,links),nominal,[0.,0.,-drop],4.,reference_only=True,input_reference=finite_input)
    return dict(available=True,goal=goal,qair=qair,normal_targets=targets,feedback=feedback,
                refinement=info,transfer=transfer,descent=descent)
