"""Redistribute thrust without changing a baseline flight wrench.

The secondary objective uses known rotor actions, not inferred contact forces.
It is a bounded, frozen-pose calculation and does not guarantee grasp success.
"""
import numpy as np
import pinocchio as pin
from scipy.linalg import null_space
from scipy.optimize import minimize, LinearConstraint
from .linear_program import linprog


def rotor_actions(model, meta, q):
    data = model.createData()
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    columns = []
    for rotor in meta['rotors']:
        frame = rotor['frame']
        jacobian = pin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)
        columns.append(jacobian[:3].T @ data.oMf[frame].rotation[:, 2])
    return np.column_stack(columns)


def preserve_wrench_distribution(base, joint, baseline, reference, maximum, effort):
    """Keep every supplied base-action row fixed; optimize only its nullspace."""
    baseline, reference, maximum = map(np.asarray, (baseline, reference, maximum))
    normalized = np.asarray(joint) / np.asarray(effort)[:, None]
    basis = null_space(np.asarray(base), rcond=1e-9)
    if not basis.shape[1]:
        return baseline.copy(), dict(accepted=False, reason='no_wrench_nullspace')
    # A small input-distance term resolves equally good joint-action solutions.
    matrix = np.vstack([normalized, .01 * np.diag(1. / maximum)])
    residual = matrix @ (baseline - reference)
    reduced = matrix @ basis
    hessian = reduced.T @ reduced
    gradient0 = reduced.T @ residual
    # Whiten this exact quadratic so SLSQP does not spend iterations learning
    # the badly scaled Hessian of torque and small tie-breaking terms.
    transform = np.linalg.solve(np.linalg.cholesky(hessian).T, np.eye(len(hessian)))
    direction = basis @ transform
    gradient = transform.T @ gradient0
    solution = minimize(lambda z: .5 * z @ z + gradient @ z,
                        np.zeros(basis.shape[1]), jac=lambda z: z + gradient,
                        method='SLSQP',
                        constraints=[LinearConstraint(direction, -baseline, maximum-baseline)],
                        options=dict(maxiter=60, ftol=1e-10))
    raw_candidate = baseline + direction @ solution.x
    candidate = np.clip(raw_candidate, 0., maximum)
    before = float(np.linalg.norm(normalized @ (baseline-reference)))
    after = float(np.linalg.norm(normalized @ (candidate-reference)))
    wrench_error = float(np.max(abs(base @ (candidate-baseline)), initial=0.))
    accepted = bool(solution.success and wrench_error < 1e-7
                    and raw_candidate.min() >= -1e-8 and np.all(raw_candidate <= maximum+1e-8)
                    and after <= before+1e-8)
    diagnostics = dict(accepted=accepted, solver_status=int(solution.status),
                       iterations=int(solution.nit), nullity=int(basis.shape[1]),
                       action_difference_before=before, action_difference_after=after,
                       wrench_change=wrench_error)
    return (candidate if accepted else baseline.copy()), diagnostics


def bound_joint_loads(base, joint, required, baseline, maximum, effort, *, load_bounds=None):
    """Closest thrust with unchanged base actions and bounded predicted loads.

    ``required - joint @ thrust`` is an inverse-dynamics prediction at supplied
    poses/accelerations. This does not measure actual drive/contact forces.
    An unsuccessful solve returns no command, never an unqualified fallback.
    """
    baseline,maximum,required,effort=map(np.asarray,(baseline,maximum,required,effort))
    base,joint=map(np.asarray,(base,joint))
    before=required-joint@baseline
    low,high=(-effort,effort) if load_bounds is None else map(np.asarray,load_bounds)
    if np.any(low>high):
        return None,dict(accepted=False,reason='incompatible_load_bounds')
    if np.all(before>=low-1e-9) and np.all(before<=high+1e-9):
        return baseline.copy(),dict(accepted=True,unchanged=True,peak_ratio=float(max(abs(before)/effort)))
    direction=5.*null_space(base,rcond=1e-9)
    if not direction.shape[1]:
        return None,dict(accepted=False,reason='no_wrench_nullspace')
    matrix=np.vstack([direction/maximum[:,None],(joint@direction)/effort[:,None]])
    lower=np.r_[-baseline/maximum,(before-high)/effort]
    upper=np.r_[(maximum-baseline)/maximum,(before-low)/effort]
    result=minimize(lambda z:.5*z@z,np.zeros(direction.shape[1]),jac=lambda z:z,
        constraints=[LinearConstraint(matrix,lower,upper)],method='SLSQP',
        options=dict(maxiter=60,ftol=1e-10))
    candidate=np.clip(baseline+direction@result.x,0.,maximum)
    loads=required-joint@candidate
    wrench_error=float(abs(base@(candidate-baseline)).max(initial=0.))
    peak=float(max(abs(loads)/effort))
    accepted=bool(result.success and wrench_error<1e-7
                  and np.all(loads>=low-1e-8) and np.all(loads<=high+1e-8))
    diagnostics=dict(accepted=accepted,solver_status=int(result.status),iterations=int(result.nit),
        peak_ratio=peak,wrench_change=wrench_error,maximum_thrust_change=float(max(abs(candidate-baseline))))
    return (candidate if accepted else None),diagnostics


def preserve_requested_action(base,joint,baseline,command,position,velocity,kp,kd,effort,limits,maximum):
    """Pair thrust and position commands to realize the same unsaturated request.

    Equality is instantaneous at the observed pose/velocity, not a dynamic
    stability guarantee. No actual contact load or acceleration is an input.
    """
    baseline,command,position,velocity,kp,kd,effort,limits=map(
        np.asarray,(baseline,command,position,velocity,kp,kd,effort,limits))
    raw=kp*(command-position)-kd*velocity
    low=np.maximum(-effort,kp*(limits[:,0]-position)-kd*velocity)
    high=np.minimum(effort,kp*(limits[:,1]-position)-kd*velocity)
    force,info=bound_joint_loads(base,joint,raw+joint@baseline,baseline,maximum,effort,
                               load_bounds=(low,high))
    if force is None:return None,None,info
    paired=command-joint@(force-baseline)/kp
    info.update(maximum_position_command_change_rad=float(abs(paired-command).max(initial=0.)))
    return paired,force,info


def redistribute_observed_load(preserved_action,joint_action,observed_motor,baseline,maximum,effort):
    """Minimize measured joint burden at frozen contact load and acceleration.

    ``preserved_action`` must include every unmodified actuator/base action.
    The returned motor target is conditional on realizing the paired input;
    thrust alone is not an admissible implementation of this calculation.
    """
    base,joint=map(np.asarray,(preserved_action,joint_action))
    motor,baseline,maximum,effort=map(np.asarray,(observed_motor,baseline,maximum,effort))
    nr=len(baseline)
    if (base.ndim!=2 or base.shape[1]!=nr or joint.shape!=(len(motor),nr)
        or maximum.shape!=(nr,) or effort.shape!=motor.shape
        or not all(np.isfinite(a).all() for a in [base,joint,motor,baseline,maximum,effort])
        or np.any(maximum<=0) or np.any(effort<=0)
        or np.any(baseline<0) or np.any(baseline>maximum)):
        raise ValueError('invalid measured-load distribution input')
    basis=null_space(base,rcond=1e-9)
    if not basis.shape[1]:
        return None,dict(accepted=False,reason='no_preserved_action_nullspace')
    direction=5.*basis;action=joint@direction
    C=np.vstack([np.column_stack([direction,np.zeros(nr)]),
                 np.column_stack([-direction,np.zeros(nr)]),
                 np.column_stack([-action,-effort]),
                 np.column_stack([action,-effort])])
    upper=np.r_[maximum-baseline,baseline,-motor,motor]
    solution=linprog(np.r_[np.zeros(basis.shape[1]),1.],A_ub=C,b_ub=upper,
        bounds=[(None,None)]*basis.shape[1]+[(0,1.)],method='highs')
    if not solution.success:
        return None,dict(accepted=False,reason='measured_load_distribution_not_found',solver_status=int(solution.status))
    peak=float(solution.x[-1]);peak_bound=min(1.,peak+1e-7)
    matrix=np.vstack([direction/maximum[:,None],action/effort[:,None]])
    lower=np.r_[-baseline/maximum,motor/effort-peak_bound]
    upper=np.r_[(maximum-baseline)/maximum,motor/effort+peak_bound]
    closest=minimize(lambda z:.5*z@z,solution.x[:-1],jac=lambda z:z,
        constraints=[LinearConstraint(matrix,lower,upper)],method='SLSQP',
        options=dict(maxiter=60,ftol=1e-10))
    candidate=baseline+direction@closest.x
    delta=candidate-baseline;target=motor-joint@delta
    action_error=float(abs(base@delta).max(initial=0.))
    valid=bool(closest.success and np.min(candidate)>=-1e-8
        and np.all(candidate<=maximum+1e-8) and action_error<=1e-7
        and np.all(abs(target)<=effort*peak_bound+1e-8))
    diagnostics=dict(accepted=valid,minimum_peak_ratio=peak,
        observed_peak_ratio=float(max(abs(motor)/effort)),
        conditional_peak_ratio=float(max(abs(target)/effort)),
        conditional_motor_target=target.tolist(),preserved_action_error=action_error,
        maximum_thrust_change=float(max(abs(delta))),secondary_success=bool(closest.success))
    return (candidate if valid else None),diagnostics
