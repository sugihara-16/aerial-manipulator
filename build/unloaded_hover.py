"""Minimum input change for gravity balance of the robot without object load.

The supported object is excluded from this robot-only equilibrium. Commands
retain real joint damping; current joint positions anchor the position gains.
This is a supported-release component, not a complete flight controller.
"""
import numpy as np
import proxsuite
from amsrr_det.linear_program import linprog


def command(local,q,previous,acceleration=None):
    local.at(q)
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
    reserve=linprog(np.r_[np.zeros(n),1.],A_eq=np.c_[A,np.zeros(len(rhs))],b_eq=rhs,
        A_ub=np.vstack([np.c_[np.eye(n),-np.ones(n)],np.c_[-np.eye(n),-np.ones(n)]]),
        b_ub=np.zeros(2*n),bounds=list(zip(lo,hi))+[(0.,1.)])
    if not reserve.success:raise ValueError('no unloaded gravity allocation inside motor bounds')
    fraction=1-.95*(1-reserve.x[-1])
    lo=np.maximum(lo,-fraction);hi=np.minimum(hi,fraction)
    solver=proxsuite.proxqp.dense.QP(n,len(rhs),n)
    solver.settings.eps_abs=solver.settings.eps_rel=1e-9
    solver.init(np.eye(n),-(previous-offset)/scale,A,rhs,np.eye(n),lo,hi)
    solver.solve();x=solver.results.x
    residual=max(float(abs(A@x-rhs).max()),float(np.max(lo-x)),float(np.max(x-hi)))
    if not str(solver.results.info.status).endswith('PROXQP_SOLVED') or residual>1e-7:
        raise ValueError(f'unloaded gravity allocation unresolved: {residual:g}')
    return offset+scale*x,dict(equilibrium_residual=residual,
        maximum_normalized_static_motor=float(abs(x).max()),
        minimum_common_motor_fraction=float(reserve.x[-1]),retained_motor_bound=float(fraction))
