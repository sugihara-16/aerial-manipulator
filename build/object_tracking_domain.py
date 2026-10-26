"""Separate object pose/rate bounds over a predicted held-input interval."""
import numpy as np,mujoco,clarabel

def cross(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def errors(model,q,v,target,velocity,D=None):
    j=model.joint('object').id;qi=int(model.jnt_qposadr[j]);vi=int(model.jnt_dofadr[j])
    angle=np.empty(3);a=np.empty(9);b=np.empty(9)
    mujoco.mju_subQuat(angle,q[qi+3:qi+7],target[qi+3:qi+7])
    mujoco.mju_quat2Mat(a,q[qi+3:qi+7]);mujoco.mju_quat2Mat(b,target[qi+3:qi+7])
    R,S=a.reshape(3,3),b.reshape(3,3)
    values=np.array([q[qi:qi+3]-target[qi:qi+3],angle,v[vi:vi+3]-velocity[vi:vi+3],R@v[vi+3:vi+6]-S@velocity[vi+3:vi+6]])
    if D is None:return values
    dq,dv=D[:model.nv],D[model.nv:];da=np.empty((3,3))
    mujoco.mjd_subQuat(q[qi+3:qi+7],target[qi+3:qi+7],da,None)
    derivatives=np.array([dq[vi:vi+3],da@dq[vi+3:vi+6],dv[vi:vi+3],R@(dv[vi+3:vi+6]-cross(v[vi+3:vi+6])@dq[vi+3:vi+6])])
    return values,derivatives


class ObjectTrackingDomain:
    def __init__(self,model,q,v,reference,velocity):
        self.m=model;self.start_target=reference.copy();self.velocity=velocity.copy()
        self.scales=np.array([.03,.2,.02,.1])
        self.allowed=np.maximum(self.scales,np.linalg.norm(errors(model,q,v,reference,velocity),axis=1))

    def samples(self,r,D=None):
        m=self.m
        for i,row in enumerate(r['states']):
            target=self.start_target.copy();mujoco.mj_integratePos(m,target,self.velocity,float(row[0]))
            yield errors(m,row[1:1+m.nq],row[1+m.nq:],target,self.velocity,None if D is None else D[i])

    def violation(self,r):
        return max(float(np.max(np.linalg.norm(e,axis=1)/self.allowed-1.)) for e in self.samples(r))

    def constraints(self,r,D,lower,upper):
        a=[];b=[];c=[]
        for values,derivatives in self.samples(r,D):
            for value,derivative,limit in zip(values,derivatives,self.allowed):
                if np.linalg.norm(value)+np.linalg.norm(derivative,axis=0)@np.maximum(abs(lower),abs(upper))>limit:
                    a.append(np.vstack([np.zeros(self.m.nu),-derivative]));b.append(np.r_[limit,value]);c.append(clarabel.SecondOrderConeT(4))
        return a,b,c
