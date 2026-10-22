"""Task-direction acceleration and feedback reserve at one generated posture.

The total rotor wrench is checked with fixed actual gimbal directions. Internal
Dock torques cancel; this necessary drive condition is not a grasp certificate.
"""
import numpy as np
from amsrr_det.affine_bounds import scalar_interval,l1_image_radius
from build.momentum_feedback import MomentumFeedback,rigid_motion_map,TRACKING_TOLERANCES


def task_wrench_margin(local,q,u,displacement,*,feedback=None,measure_feedback=True):
    feedback=feedback or MomentumFeedback(local,q,u,period=.01)
    local.at(q);mass,B,_=local.matrices();T=rigid_motion_map(local.m,q)
    physical=T.T@B
    if np.max(abs(physical[:,:local.nj]))>1e-7:raise ValueError('internal joint creates external wrench')
    matrix=physical[:,local.nj:]*local.rotor_max
    lower=(local.rotor_min-u[local.nj:])/local.rotor_max
    upper=(local.rotor_max-u[local.nj:])/local.rotor_max
    inertia=T.T@mass@T;rows=[];centers=[]
    for name,delta,duration in [('lift',np.array([0.,0.,.15]),4.),
                                ('carry',np.asarray(displacement),6.)]:
        distance=float(np.linalg.norm(delta))
        if distance<1e-12:continue
        direction=np.r_[delta/distance,np.zeros(3)]
        peak=float(10/np.sqrt(3)*distance/duration**2)
        attainable=scalar_interval(matrix,np.zeros(6),inertia@direction,lower,upper)
        feasible=bool(attainable is not None and attainable[0]<=-peak and attainable[1]>=peak)
        rows.append(dict(phase=name,direction=direction[:3].tolist(),distance_m=distance,
            duration_s=duration,required_acceleration_m_s2=peak,
            attainable_acceleration_m_s2=attainable,nominal_feasible=feasible))
        centers.extend([sign*peak*(inertia@direction) for sign in (-1.,1.)])
    feasible=all(row['nominal_feasible']for row in rows)
    radius=None
    if feasible and measure_feedback:
        F=-inertia@feedback.outer_gain*TRACKING_TOLERANCES
        radius=l1_image_radius(matrix,centers,F,lower,upper)
    return dict(scope=__doc__,nominal_feasible=feasible,phases=rows,
        normalized_feedback_l1_radius=radius,feedback_tolerances=TRACKING_TOLERANCES.tolist(),
        neighborhood_robustness_certified=False,contact_maintenance_certified=False)


def refine_task_reference(plant,full_model,q,u,velocity,links,displacement,finite):
    """Repair direction only when needed, then recheck the actual contact pose."""
    from build.sticking_design import LocalGrasp
    from build.task_gimbal_design import GimbalWrenchDesign
    from build.finite_slip_preload import construct_finite_slip
    records=[]
    for iteration in range(4):
        local=LocalGrasp(plant,q,links)
        margin=task_wrench_margin(local,q,u,displacement,measure_feedback=False)
        if margin['nominal_feasible']:
            margin=task_wrench_margin(local,q,u,displacement)
            return (q,u,velocity),finite,dict(margin=margin,repairs=records)
        if iteration==3:break
        last=finite['iterations'][-1]['recheck']if finite['iterations']else finite['initial_fixed']
        pose,change=GimbalWrenchDesign(local,q,displacement).solve(u,.95*last['reserve'])
        records.append(dict(before=margin,angle_design=change))
        if not change['accepted']:break
        (q,u,velocity),finite=construct_finite_slip(plant,full_model,pose,links,23.,minimize_input=True)
    error=ValueError('task acceleration unavailable after fixed direction refinement')
    error.evidence=dict(margin=margin,repairs=records)
    raise error
