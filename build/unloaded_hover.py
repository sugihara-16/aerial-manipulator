"""Minimum input change for gravity balance of the robot without object load.

The supported object is excluded from this robot-only equilibrium. Commands
retain real joint damping; current joint positions anchor the position gains.
This is a supported-release component, not a complete flight controller.
"""
import numpy as np
import proxsuite


def _closest_input(A,rhs,lo,hi,target,workspace=None):
    n=len(target)
    solver=None if workspace is None else getattr(workspace,'_hover_qp',None)
    if solver is None:
        solver=proxsuite.proxqp.dense.QP(n,len(rhs),n)
        solver.init(np.eye(n),-target,A,rhs,np.eye(n),lo,hi)
        if workspace is not None:workspace._hover_qp=solver
    else:solver.update(g=-target,A=A,b=rhs,l=lo,u=hi)
    solver.settings.eps_abs=solver.settings.eps_rel=1e-9
    for tolerance in [1e-9,1e-12]:
        solver.settings.eps_abs=solver.settings.eps_rel=tolerance
        solver.solve();x=solver.results.x
        residual=max(float(abs(A@x-rhs).max()),float(np.max(lo-x)),float(np.max(x-hi)))
        if str(solver.results.info.status).endswith('PROXQP_SOLVED') and residual<=1e-7:
            return x.copy(),residual
    raise ValueError(f'unloaded gravity allocation unresolved: {residual:g}')


def command(local,q,previous,acceleration=None):
    local.mechanical_state(q)
    M,B,bias=local.matrices();m=local.m;n=B.shape[1]
    keep=np.ones(m.nv,bool);obj=m.jnt_dofadr[m.joint('object').id];keep[obj:obj+6]=False
    offset=np.r_[np.where(local.dock,local.kp*q[local.qi],q[local.qi]),np.zeros(len(local.sites))]
    scale=np.r_[local.effort/local.gain,local.rotor_max]
    A=(B*scale)[keep];rhs=(-bias-B@offset)[keep]
    if acceleration is not None:rhs+=(M@acceleration)[keep]
    lo=np.r_[-np.ones(local.nj),local.rotor_min/local.rotor_max]
    hi=np.ones(n)
    limited=m.actuator_ctrllimited.astype(bool)
    lo[limited]=np.maximum(lo[limited],(m.actuator_ctrlrange[limited,0]-offset[limited])/scale[limited])
    hi[limited]=np.minimum(hi[limited],(m.actuator_ctrlrange[limited,1]-offset[limited])/scale[limited])
    gimbal=np.flatnonzero(~local.dock)
    lo[gimbal]=np.maximum(lo[gimbal],(m.jnt_range[local.joint_ids[gimbal],0]-offset[gimbal])/scale[gimbal])
    hi[gimbal]=np.minimum(hi[gimbal],(m.jnt_range[local.joint_ids[gimbal],1]-offset[gimbal])/scale[gimbal])
    # Retain 95% of the maximum common reserve, as in the existing static
    # design. Equivalent gravity allocations must leave room for real damping.
    from amsrr_det.repeated_lp import RepeatedLinearProgram
    if not hasattr(local,'_hover_reserve'):local._hover_reserve=RepeatedLinearProgram()
    reserve=local._hover_reserve
    reserve.load(np.r_[np.zeros(n),1.],
        np.vstack([np.c_[A,np.zeros(len(rhs))],np.c_[np.eye(n),-np.ones(n)],np.c_[-np.eye(n),-np.ones(n)]]),
        np.r_[rhs,np.full(2*n,-np.inf)],np.r_[rhs,np.zeros(2*n)],np.r_[lo,0.],np.r_[hi,1.])
    optimum=reserve.solve()
    if optimum is None:raise ValueError('no unloaded gravity allocation inside motor bounds')
    fraction=1-.95*(1-optimum[-1])
    lo=np.maximum(lo,-fraction);hi=np.minimum(hi,fraction)
    x,residual=_closest_input(A,rhs,lo,hi,(previous-offset)/scale,workspace=local)
    return offset+scale*x,dict(equilibrium_residual=residual,
        maximum_normalized_static_motor=float(abs(x).max()),
        minimum_common_motor_fraction=float(optimum[-1]),retained_motor_bound=float(fraction))
