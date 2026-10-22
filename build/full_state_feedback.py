"""Generate a local full-state return law once from known held-input mechanics.

Online evaluation is a matrix product and the existing physical command bounds.
This local model is not a whole-task certificate; independent execution audits
and an experimentally checked entry neighborhood remain necessary.
"""
import time
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are
from build.finite_difference_response import FiniteDifferenceResponse


def observed_tangent_speed(m,d,links):
    """Relative material speed from observed geometry and velocity, not force."""
    obj=m.geom('object').id;maximum=0.
    for c in d.contact:
        ga,gb=map(int,c.geom)
        if c.dist>0 or obj not in (ga,gb):continue
        other=gb if ga==obj else ga
        if not any(m.geom(other).name.startswith(link+'/') for link in links):continue
        ja=np.zeros((3,m.nv));jb=ja.copy()
        mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[ga]))
        mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[gb]))
        maximum=max(maximum,float(np.linalg.norm((c.frame.reshape(3,3)@(jb-ja)@d.qvel)[1:])))
    return maximum


def design_feedback(plant,links,q,u,details=None,*,state_step=1e-5,input_step=1e-4,normal_margin_cost=False,period=.01,nominal_velocity=None):
    """Full position/velocity differential around the supplied nominal state."""
    began=time.perf_counter();m=plant.mj;nv=m.nv
    steps=round(period/m.opt.timestep)
    if steps<1 or abs(steps*m.opt.timestep-period)>1e-10:raise ValueError('feedback period must contain whole physical steps')
    response=FiniteDifferenceResponse(plant,links,relative_step=input_step,period=period,steps=steps)
    v=np.zeros(nv)if nominal_velocity is None else np.asarray(nominal_velocity).copy()
    if v.shape!=(nv,):raise ValueError('nominal velocity dimension mismatch')
    base=response.evaluate(q,v,u,True)
    B=base['derivatives'][-1]
    ps=np.ones(nv);vs=np.full(nv,3.)
    for j in range(m.njnt):
        vi=int(m.jnt_dofadr[j])
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            ps[vi:vi+6]=[.03]*3+[.2]*3;vs[vi:vi+6]=[.02]*3+[.1]*3
        else:ps[vi]=max(float(np.diff(m.jnt_range[j])[0])/2,.01)
    xs=np.r_[ps,vs];us=response.scale

    def derivative(i):
        epsilon=state_step*xs[i];qp=q.copy();qm=q.copy();vp=v.copy();vm=v.copy()
        if i<nv:
            direction=np.zeros(nv);direction[i]=1.
            mujoco.mj_integratePos(m,qp,direction,epsilon)
            mujoco.mj_integratePos(m,qm,direction,-epsilon)
        else:vp[i-nv]+=epsilon;vm[i-nv]-=epsilon
        plus=response._response((qp,vp,u));minus=response._response((qm,vm,u))
        column=np.empty(2*nv)
        mujoco.mj_differentiatePos(m,column[:nv],2*epsilon,minus['q'],plus['q'])
        column[nv:]=(plus['v']-minus['v'])/(2*epsilon)
        return column

    A=np.column_stack(list(response.pool.map(derivative,range(2*nv))))
    As=A*xs[None,:]/xs[:,None];Bs=B*us[None,:]/xs[:,None]
    Q=np.eye(2*nv)
    if normal_margin_cost:
        # A pose error is not equally harmless in every direction. Normalize
        # predicted pressure change by the remaining physical load reserve,
        # using the same pad law and limits as static generation. This is an
        # objective term, not a nonlinear safety certificate.
        from build.sticking_design import LocalGrasp
        local=LocalGrasp(plant,q,links);gap,_,Jn,_=local.contact()
        pads=np.array([e[0]for e in local.elements]);pressure=-local.k*gap
        load=np.array([pressure[pads==i].sum()for i in range(len(links))])
        margin=np.minimum(load-.5,30./np.sqrt(1+4.5**2)-load)
        if min(margin)<=0:raise ValueError('no normal-load reserve at the local design')
        output=np.zeros((len(links),2*nv))
        for i in range(len(links)):output[i,:nv]=-local.k*Jn[pads==i].sum(axis=0)*xs[:nv]/margin[i]
        Q+=output.T@output
    if details is not None:details.update(A=As,B=Bs,Q=Q,state_scale=xs,input_scale=us)
    try:P=solve_discrete_are(As,Bs,Q,np.eye(m.nu))
    except np.linalg.LinAlgError:
        response.close();raise
    Ks=np.linalg.solve(np.eye(m.nu)+Bs.T@P@Bs,Bs.T@P@As)
    K=us[:,None]*Ks/xs[None,:]
    drift=np.empty(2*nv);mujoco.mj_differentiatePos(m,drift[:nv],1.,q,base['q']);drift[nv:]=base['v']-v
    ds=drift/xs;fixed=np.linalg.solve(np.eye(2*nv)-As+Bs@Ks,ds)
    report=dict(period_s=period,nominal_velocity_norm=float(np.linalg.norm(v)),state_difference_step=state_step,input_difference_step=input_step,preparation_s=time.perf_counter()-began,closed_loop_radius=float(abs(np.linalg.eigvals(As-Bs@Ks)).max()),
                nominal_drift_scaled=float(np.linalg.norm(ds)),predicted_steady_error_scaled=float(np.linalg.norm(fixed)),
                predicted_steady_input_scaled=float(np.max(abs(Ks@fixed))),state_dimension=2*nv,input_dimension=m.nu,normal_margin_cost=normal_margin_cost)
    if normal_margin_cost:report.update(nominal_normal_n=load.tolist(),normal_reserve_n=margin.tolist())
    if details is not None:details.update(F=As-Bs@Ks,A=As,B=Bs,P=P,state_scale=xs,drift=ds,K_scaled=Ks,input_scale=us)
    response.close()
    return K,report


def tracking_feedforward(m,linear,references,period=.01):
    """Finite-horizon affine LQ tracking with the same terminal Riccati cost.

    Common translations are Galilean invariant away from support contacts.
    Reference acceleration and the nominal model drift are explicit forcing;
    no rollout outcomes choose the path. Bounds are still checked by update.
    """
    F,B,P=linear['F'],linear['B'],linear['P'];xs=linear['state_scale']
    solve=np.linalg.solve(np.eye(m.nu)+B.T@P@B,B.T)
    costate=np.zeros(2*m.nv);offsets=np.empty((len(references)-1,m.nu));increment=np.zeros(m.nv)
    for k in range(len(offsets)-1,-1,-1):
        q,v=references[k];qn,vn=references[k+1]
        mujoco.mj_differentiatePos(m,increment,1.,q,qn)
        forcing=linear['drift']+np.r_[period*v-increment,v-vn]/xs
        value=P@forcing+costate
        offsets[k]=-linear['input_scale']*(solve@value)
        costate=F.T@value
    return offsets


def rigid_object_gain(model,q,K):
    """Restrict outer correction to a common rigid motion of the whole grasp.

    This changes only the feedback observation. Internal errors and physical
    motor limits are still evaluated at the actual state; no state is reset.
    It is a diagnostic of outer-loop amplification, not a grasp certificate.
    """
    object_joint=model.joint('object').id
    oi=int(model.jnt_qposadr[object_joint]);ov=int(model.jnt_dofadr[object_joint])
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,q[oi+3:oi+7]);rotation=rotation.reshape(3,3)
    lift=np.zeros((model.nv,6))
    for j in np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
        qi=int(model.jnt_qposadr[j]);vi=int(model.jnt_dofadr[j]);r=q[qi:qi+3]-q[oi:oi+3]
        cross=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
        frame=np.empty(9);mujoco.mju_quat2Mat(frame,q[qi+3:qi+7])
        lift[vi:vi+3,:3]=np.eye(3);lift[vi:vi+3,3:]=-cross@rotation
        lift[vi+3:vi+6,3:]=frame.reshape(3,3).T@rotation
    restricted=np.zeros_like(K)
    restricted[:,ov:ov+6]=K[:,:model.nv]@lift
    restricted[:,model.nv+ov:model.nv+ov+6]=K[:,model.nv:]@lift
    return restricted


class FullStateFeedback:
    def _command_state(self,local,q,u):
        self.local=local;self.m=local.m;self.u=u.copy();self.nominal=q.copy()
        self.error=np.zeros(self.m.nv);self.clipped_periods=0
        self.rigid_object_only=False
        self.gain=self.m.actuator_gainprm[:,0].copy()

    @classmethod
    def from_generated(cls,local,q,u,K):
        """Evaluate an already generated gain without doing dynamics synthesis."""
        instance=cls.__new__(cls);instance._command_state(local,q,u)
        if K.shape!=(local.m.nu,2*local.m.nv):raise ValueError('generated gain dimension mismatch')
        instance.K=K.copy();instance.info={};instance.linearization={}
        return instance

    def __init__(self,local,q,u=None,*,normal_margin_cost=False,period=.01,nominal_velocity=None):
        if u is None:
            u,info=local.equilibrium(q)
            if not info['feasible']:raise ValueError('full feedback equilibrium unavailable')
        self._command_state(local,q,u);self.linearization={}
        self.K,self.info=design_feedback(local.plant,local.links,q,u,self.linearization,normal_margin_cost=normal_margin_cost,period=period,nominal_velocity=nominal_velocity)

    def update(self,q,v,reference,velocity=None,input_reference=None):
        local=self.local;m=self.m;nj=local.nj
        mujoco.mj_differentiatePos(m,self.error,1.,reference,q)
        K=rigid_object_gain(m,reference,self.K)if self.rigid_object_only else self.K
        command=(self.u if input_reference is None else input_reference)-K@np.r_[self.error,v if velocity is None else v-velocity]
        offset=local.kp*q[local.qi]+local.kd*v[local.vi]
        low=np.r_[(offset-local.effort)/self.gain[:nj],local.rotor_min]
        high=np.r_[(offset+local.effort)/self.gain[:nj],local.rotor_max]
        gimbal=~local.dock
        low[:nj][gimbal]=np.maximum(low[:nj][gimbal],m.jnt_range[local.joint_ids[gimbal],0])
        high[:nj][gimbal]=np.minimum(high[:nj][gimbal],m.jnt_range[local.joint_ids[gimbal],1])
        limited=m.actuator_ctrllimited.astype(bool)
        low[limited]=np.maximum(low[limited],m.actuator_ctrlrange[limited,0]);high[limited]=np.minimum(high[limited],m.actuator_ctrlrange[limited,1])
        if np.any(low>high):raise ValueError('empty_command_domain')
        self.clipped_periods+=int(np.any((command<low)|(command>high)))
        return np.clip(command,low,high)
