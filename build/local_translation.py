"""A rigid translation reference from the SAME local mechanics.

Affine interpolation of two feasible acceleration allocations preserves the
linear force balance, actuator bounds and convex friction bounds at this pose.
It is conditional on the stated local contact model; execution is independent.
No simulated trajectory is optimized or selected here.
"""
import numpy as np
import mujoco


class LocalTranslation:
    def __init__(self,design,q,displacement,duration,*,reference_only=False):
        self.design=design;self.q=q.copy();self.delta=np.asarray(displacement);self.duration=duration
        self.free=np.flatnonzero(design.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        self.direction=np.zeros(design.m.nv)
        for j in self.free:
            v=design.m.jnt_dofadr[j];self.direction[v:v+3]=self.delta
        self.peak=10/np.sqrt(3)/duration**2
        self.minus,negative=design.equilibrium(q,-self.peak*self.direction)
        self.plus,positive=design.equilibrium(q,self.peak*self.direction)
        if not negative['feasible'] or not positive['feasible']:raise ValueError('translation acceleration outside current force domain')
        design.at(q)
        for command,allocation in [(self.minus,negative),(self.plus,positive)]:
            loads=np.array(allocation['normal_force']);tangent=np.array(allocation['tangent_force']).reshape(-1,2)
            forces=np.zeros((len(design.links),3));moments=forces.copy();normal=np.zeros(len(design.links))
            for i,(pad,bp,bo,lp,_,lf)in enumerate(design.elements):
                frame=lf@design.d.xmat[bo].reshape(3,3).T
                force=frame.T@np.r_[loads[i],tangent[i]];arm=design.d.xmat[bp].reshape(3,3)@lp
                forces[pad]+=force;moments[pad]+=np.cross(arm,force);normal[pad]+=loads[i]
            gimbals=~design.dock;limits=design.m.jnt_range[design.joint_ids[gimbals]]
            if (max(np.linalg.norm(forces,axis=1))>30. or max(np.linalg.norm(moments,axis=1))>5. or min(normal)<.5 or
                np.any(command[:design.nj][gimbals]<limits[:,0])or np.any(command[:design.nj][gimbals]>limits[:,1])):
                raise ValueError('translation acceleration outside current force domain')
        self.center=(self.minus+self.plus)/2
        if not reference_only:self.feedback=design.linear_feedback(q,self.center)
        self.info=dict(method='fixed_deformation_feasible',scope=__doc__,duration_s=duration,displacement_m=self.delta.tolist(),
                       negative=negative,positive=positive)

    def reference(self,time):
        r=np.clip((time-2.)/self.duration,0.,1.)
        s=10*r**3-15*r**4+6*r**5
        ds=(30*r*r-60*r**3+30*r**4)/self.duration
        dds=(60*r-180*r*r+120*r**3)/self.duration**2
        q=self.q.copy()
        for j in self.free:
            qi=self.design.m.jnt_qposadr[j];q[qi:qi+3]+=s*self.delta
        fraction=(1+dds/self.peak)/2
        return q,ds*self.direction,(1-fraction)*self.minus+fraction*self.plus,s

    def update(self,q,v,time):
        reference,velocity,u,_=self.reference(time);delta=np.zeros(self.design.m.nv)
        mujoco.mj_differentiatePos(self.design.m,delta,1.,reference,q)
        N,K=self.feedback['N'],self.feedback['K']
        return u-K@np.r_[N.T@delta,N.T@(v-velocity)]


def generated_translation(design,q,displacement,duration):
    """Use a feasible restricted solution before the general local solve.

    Both branches use the same mechanics. This chooses no branch by shape ID
    or simulated success. Failure of the restricted problem is not physical
    impossibility; the deformation-aware problem retains those freedoms.
    """
    try:return LocalTranslation(design,q,displacement,duration)
    except ValueError as error:
        if str(error)!='translation acceleration outside current force domain':raise
    return PredictiveTranslation(design,q,displacement,duration)


class PredictiveTranslation:
    """Same spatial reference, with dynamically generated local deformation."""
    def __init__(self,design,q,displacement,duration,*,reference_only=False,input_reference=None):
        self.design=design;self.q=q.copy();self.delta=np.asarray(displacement);self.duration=duration
        self.free=np.flatnonzero(design.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        self.direction=np.zeros(design.m.nv)
        for j in self.free:
            v=design.m.jnt_dofadr[j];self.direction[v:v+3]=self.delta
        if reference_only:
            # The interval controller needs the path and equilibrium input,
            # not an unused second controller and its linearized dynamics.
            if input_reference is None:
                self.center,report=design.equilibrium(q)
                if not report['feasible']:raise ValueError('translation reference requires a nominal equilibrium')
            else:self.center=np.asarray(input_reference).copy()
            self.mpc=None
            self.info=dict(method='finite_slip_reference'if input_reference is not None else'equilibrium_reference',duration_s=duration,displacement_m=self.delta.tolist())
        else:
            from build.local_contact_mpc import LocalContactMPC
            self.mpc=LocalContactMPC(design,q,fast_solve=True)
            self.center=self.mpc.u;self.feedback=self.mpc.feedback
            self.info=dict(method='deformation_prediction',scope='Local compliant/sticking prediction; independent physical execution',duration_s=duration,
                           displacement_m=self.delta.tolist(),preparation_s=self.mpc.preparation_s)

    def reference(self,time):
        r=np.clip((time-2.)/self.duration,0.,1.);s=10*r**3-15*r**4+6*r**5
        ds=(30*r*r-60*r**3+30*r**4)/self.duration
        q=self.q.copy()
        for j in self.free:
            qi=self.design.m.jnt_qposadr[j];q[qi:qi+3]+=s*self.delta
        return q,ds*self.direction,self.center,s

    def update(self,q,v,time):
        if self.mpc is None:raise ValueError('reference-only path has no controller')
        references=[self.reference(time+(k+1)*self.mpc.h)[:2]for k in range(self.mpc.H)]
        return self.mpc.update(q,v,references)
