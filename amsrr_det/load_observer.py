"""Joint external-load estimate from observed motion and past motor commands.

A generalized load is not an individual pad-force measurement. The estimate
includes model/discretization errors and is delayed; it must be validated at
contact transitions before being used as an allocation constraint.
"""
import numpy as np
import pinocchio as pin


class JointLoadObserver:
    def __init__(self,model,meta,period=.01,filter_time=.02):
        self.model,self.meta,self.dt=model,meta,period;self.data=model.createData()
        self.actuators=meta['actuators'];self.vi=np.array([a['v'] for a in self.actuators])
        self.qi=np.array([a['q'] for a in self.actuators])
        self.kp=np.array([a['kp'] for a in self.actuators]);self.kd=np.array([a['kd'] for a in self.actuators])
        self.limits=np.array([a['effort'] for a in self.actuators])
        self.alpha=1. if filter_time==0 else 1-np.exp(-period/filter_time)
        self.previous=None;self.external=None;self.raw=None

    def rotor_load(self,q,forces):
        m,d=self.model,self.data;pin.framesForwardKinematics(m,d,q)
        external=[pin.Force.Zero() for _ in range(m.njoints)]
        for rotor,force in zip(self.meta['rotors'],forces):
            frame=m.frames[rotor['frame']]
            external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,force]),np.zeros(3)))
        zero=np.zeros(m.nv)
        # Difference of inverse dynamics isolates only the rotor generalized load.
        plain=pin.rnea(m,d,q,zero,zero).copy()
        return plain-pin.rnea(m,d,q,zero,zero,external)

    def update(self,q,v,past_command,past_forces):
        if self.previous is None:
            self.previous=(q.copy(),v.copy());return None
        oldq,oldv=self.previous;self.previous=(q.copy(),v.copy())
        midpoint=pin.interpolate(self.model,oldq,q,.5);velocity=(oldv+v)/2
        acceleration=(v-oldv)/self.dt
        required=pin.rnea(self.model,self.data,midpoint,velocity,acceleration).copy()
        torque=np.clip(self.kp*(np.asarray(past_command)-midpoint[self.qi])-self.kd*velocity[self.vi],-self.limits,self.limits)
        residual=required-self.rotor_load(midpoint,past_forces);residual[self.vi]-=torque
        self.raw=residual
        self.external=residual.copy() if self.external is None else self.external+self.alpha*(residual-self.external)
        return self.external.copy()
