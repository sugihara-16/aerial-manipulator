"""Contact-free prediction with the same bounded position drives.

Backward Euler eliminates unsaturated PD torques into positive joint inertia.
An active set enforces the actual torque limits. No contact force is inferred.
This predictor is experimental and is not a task acceptance test.
An optional generalized load is held fixed in the model's velocity coordinates
for one interval. It is an approximation, not a contact-force guarantee.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import lsq_linear


class DrivePrediction:
    def __init__(self,model,meta):
        # Added inertia belongs only to this predictor, never the plant model.
        self.model=model.copy();self.data=self.model.createData();self.meta=meta
        self.qi=np.array([a['q'] for a in meta['actuators']]);self.vi=np.array([a['v'] for a in meta['actuators']])
        self.kp=np.array([a['kp'] for a in meta['actuators']]);self.kd=np.array([a['kd'] for a in meta['actuators']])
        self.limits=np.array([a['effort'] for a in meta['actuators']]);self.armature=self.model.armature.copy()

    def external_forces(self,force):
        model=self.model;external=[pin.Force.Zero() for _ in range(model.njoints)]
        for rotor,f in zip(self.meta['rotors'],force):
            frame=model.frames[rotor['frame']]
            external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
        return external

    def step(self,q,v,command,force,h=.001, *, linearize=False,external=None,generalized_load=None):
        model,data=self.model,self.data
        if external is None:external=self.external_forces(force)
        load=np.zeros(model.nv) if generalized_load is None else np.asarray(generalized_load)
        if load.shape!=(model.nv,) or not np.isfinite(load).all():raise ValueError('invalid generalized load')
        active=np.zeros(len(self.vi),dtype=int)
        bias0=self.kp*(np.asarray(command)-q[self.qi]-h*v[self.vi])-self.kd*v[self.vi]
        coefficient0=self.kp*h*h+self.kd*h
        derivatives={}
        try:
            for iteration in range(12):
                coefficient=np.where(active,0.,coefficient0)
                bias=np.where(active,active*self.limits,bias0)
                model.armature=self.armature.copy();model.armature[self.vi]+=coefficient
                torque=load.copy();torque[self.vi]+=bias
                acceleration=pin.aba(model,data,q,v,torque,external).copy()
                velocity=v+h*acceleration
                raw=self.kp*(np.asarray(command)-q[self.qi]-h*velocity[self.vi])-self.kd*velocity[self.vi]
                needed=np.where(raw>self.limits,1,np.where(raw<-self.limits,-1,0))
                if np.array_equal(active,needed):break
                active=needed
            else:
                # Coupled axes can cycle when all saturation states switch at
                # once. The same monotone drive equation has a strictly convex
                # box-constrained torque formulation; solve it without clipping
                # an inconsistent acceleration after the fact.
                model.armature=self.armature.copy()
                mass=pin.crba(model,data,q).copy();inverse=np.linalg.inv(mass)
                free_acceleration=pin.aba(model,data,q,v,load,external).copy()
                hessian=inverse[np.ix_(self.vi,self.vi)]+np.diag(1/coefficient0)
                gradient=free_acceleration[self.vi]-bias0/coefficient0
                factor=np.linalg.cholesky(hessian)
                solved=lsq_linear(factor.T,np.linalg.solve(factor,-gradient),bounds=(-self.limits,self.limits),method='bvls',tol=1e-12,max_iter=100)
                active=np.where(solved.x>=self.limits-1e-9,1,np.where(solved.x<=-self.limits+1e-9,-1,0))
                coefficient=np.where(active,0.,coefficient0);bias=np.where(active,active*self.limits,bias0)
                model.armature=self.armature.copy();model.armature[self.vi]+=coefficient
                torque=load.copy();torque[self.vi]+=bias
                acceleration=pin.aba(model,data,q,v,torque,external).copy();velocity=v+h*acceleration
                raw=self.kp*(np.asarray(command)-q[self.qi]-h*velocity[self.vi])-self.kd*velocity[self.vi]
            if linearize:
                aq,av,inverse=map(np.array,pin.computeABADerivatives(model,data,q,v,torque,external))
                free=active==0
                aq[:,self.vi]-=inverse[:,self.vi]*(self.kp*free)[None,:]
                av[:,self.vi]-=inverse[:,self.vi]*((self.kp*h+self.kd)*free)[None,:]
                pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
                action=np.column_stack([pin.getFrameJacobian(model,data,r['frame'],pin.LOCAL)[2] for r in self.meta['rotors']])
                inputs=np.column_stack([inverse[:,self.vi]*(self.kp*free)[None,:],inverse@action])
                derivatives=dict(acceleration_wrt_q=aq,acceleration_wrt_v=av,acceleration_wrt_input=inputs)
        finally:model.armature=self.armature.copy()
        applied=bias-coefficient*acceleration[self.vi]
        expected=np.clip(raw,-self.limits,self.limits)
        if not np.all(abs(applied-expected)<=1e-10+1e-10*abs(expected)):
            raise ValueError('drive_prediction_torque_inconsistent')
        return pin.integrate(model,q,h*velocity),velocity,dict(torque=applied,acceleration=acceleration,iterations=iteration+1,**derivatives)

    def interval(self,q,v,command,force,period=.01,step=.001, *, linearize=False,history=False,generalized_load=None):
        count=round(period/step)
        if count<1 or abs(count*step-period)>1e-12:raise ValueError('prediction interval must contain whole steps')
        if linearize:
            position_jac=np.zeros((self.model.nv,len(command)+len(force)));velocity_jac=position_jac.copy()
        samples=[];external=self.external_forces(force);mean_torque=np.zeros(len(command))
        for _ in range(count):
            before=q;q,v,info=self.step(q,v,command,force,step,linearize=linearize,external=external,generalized_load=generalized_load)
            mean_torque+=info['torque']/count
            if linearize:
                derivative=info['acceleration_wrt_q']@position_jac+info['acceleration_wrt_v']@velocity_jac+info['acceleration_wrt_input']
                velocity_jac=velocity_jac+step*derivative
                first,last=pin.dIntegrate(self.model,before,step*v)
                position_jac=first@position_jac+last@(step*velocity_jac)
            if history:
                sample=dict(q=q.copy(),v=v.copy())
                if linearize:sample.update(position_wrt_input=position_jac.copy(),velocity_wrt_input=velocity_jac.copy())
                samples.append(sample)
        if linearize:info.update(position_wrt_input=position_jac,velocity_wrt_input=velocity_jac)
        info['mean_torque']=mean_torque
        if history:info['samples']=samples
        return q,v,info
