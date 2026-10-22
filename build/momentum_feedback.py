"""Outer correction from whole-system momentum and physical input geometry.

No contact derivative, fitted response or force observation is used. The
allocation minimizes mass-weighted relative forcing for a prescribed whole
wrench. That objective is not a bound on contact forces or internal motion:
the supplied nominal and joint servos must independently preserve the grasp.
Ground support remains an external disturbance, not a fictitious control.
"""
import time
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are, solve_triangular
from build.full_state_feedback import FullStateFeedback

TRACKING_TOLERANCES=np.r_[[.03]*3,[.2]*3,[.02]*3,[.1]*3]


def cross_matrix(r):
    return np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])


def rigid_motion_map(model,q):
    """World twist about the object origin -> all free-body velocities."""
    oi=int(model.jnt_qposadr[model.joint('object').id])
    result=np.zeros((model.nv,6))
    for j in np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
        qi=int(model.jnt_qposadr[j]);vi=int(model.jnt_dofadr[j]);rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation,q[qi+3:qi+7])
        result[vi:vi+3,:3]=np.eye(3)
        result[vi:vi+3,3:]=-cross_matrix(q[qi:qi+3]-q[oi:oi+3])
        result[vi+3:vi+6,3:]=rotation.reshape(3,3).T
    return result


class MomentumFeedback(FullStateFeedback):
    def __init__(self,local,q,u=None,*,period=.002):
        began=time.perf_counter()
        if u is None:
            u,equilibrium=local.equilibrium(q)
            if not equilibrium['feasible']:raise ValueError('momentum feedback equilibrium unavailable')
        self._command_state(local,q,u)
        self.K=np.zeros((self.m.nu,2*self.m.nv))
        self.query=mujoco.MjData(self.m)
        self.oi=int(self.m.jnt_qposadr[self.m.joint('object').id])
        local.at(q)
        M,B,_=local.matrices()
        T=rigid_motion_map(self.m,q)
        inertia=T.T@M@T
        self.inertia=inertia
        # Only rotor thrust and supported Dock torque offsets are corrected.
        # Gimbal position commands remain on the nominal reference; no instant
        # change of their thrust direction is presumed by the allocation.
        active=np.r_[np.flatnonzero(local.dock),np.arange(local.nj,self.m.nu)]
        scale=np.r_[local.effort/local.gain,local.rotor_max]
        physical=B[:,active]*scale[active]
        white=solve_triangular(np.linalg.cholesky(M),physical,lower=True)
        _,singular,vt=np.linalg.svd(white,full_matrices=False)
        rank=np.count_nonzero(singular>singular[0]*1e-10)
        inverse=vt[:rank].T/singular[:rank]
        wrench=T.T@physical@inverse
        if np.linalg.matrix_rank(wrench,tol=np.linalg.norm(wrench)*1e-10)!=6:
            raise ValueError('fixed-gimbal whole-wrench map has insufficient rank')
        # Minimum kinetic forcing subject to the prescribed total wrench.
        # The pseudoinverse also selects minimum normalized input among
        # indistinguishable generalized forces; it selects no rollout outcome.
        allocation=inverse@np.linalg.pinv(wrench,rcond=1e-10)@inertia
        self.allocation=np.zeros((self.m.nu,6))
        self.allocation[active]=scale[active,None]*allocation
        balance=T.T@B@self.allocation-inertia
        residual=B@self.allocation-M@T
        whitened_residual=solve_triangular(np.linalg.cholesky(M),residual,lower=True)
        # Rigid-mode tracking cost uses the existing task tolerances and
        # rated input scales. This is a control design, not a rigid constraint
        # imposed on the actual multibody/contact dynamics.
        A=np.eye(12);A[:6,6:]=period*np.eye(6)
        D=np.r_[.5*period**2*np.eye(6),period*np.eye(6)]
        Q=np.diag(TRACKING_TOLERANCES**-2)
        normalized=self.allocation/scale[:,None]
        R=normalized.T@normalized
        P=solve_discrete_are(A,D,Q,R)
        self.outer_gain=np.linalg.solve(R+D.T@P@D,D.T@P@A)
        self.info=dict(method='whole_momentum_geometry',scope=__doc__,period_s=period,
            preparation_s=time.perf_counter()-began,
            wrench_balance_max=float(abs(balance).max()),
            relative_forcing_metric=(whitened_residual.T@whitened_residual).tolist(),
            rigid_model_radius=float(max(abs(np.linalg.eigvals(A-D@self.outer_gain)))),
            instantaneous_allocation_only=True)
        self.linearization={}

    def momentum(self,q,v):
        """Includes actual joint and relative-body momentum, without collision."""
        m,d=self.m,self.query
        d.qpos[:]=q
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_crb(m,d)
        mass=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
        return rigid_motion_map(m,q).T@mass@v

    def update(self,q,v,reference,velocity=None,input_reference=None):
        nominal_velocity=np.zeros(self.m.nv)if velocity is None else velocity
        error=np.empty(6);oi=self.oi
        error[:3]=q[oi:oi+3]-reference[oi:oi+3]
        body_rotation=np.empty(3);rotation=np.empty(9)
        mujoco.mju_subQuat(body_rotation,q[oi+3:oi+7],reference[oi+3:oi+7])
        mujoco.mju_quat2Mat(rotation,reference[oi+3:oi+7])
        error[3:]=rotation.reshape(3,3)@body_rotation
        momentum_error=self.momentum(q,v)-self.momentum(reference,nominal_velocity)
        equivalent_velocity=np.linalg.solve(self.inertia,momentum_error)
        correction=-self.allocation@(self.outer_gain@np.r_[error,equivalent_velocity])
        nominal=self.u if input_reference is None else input_reference
        return super().update(q,v,reference,nominal_velocity,nominal+correction)
