"""Preserve contact through constrained short held-input predictions.

Only observed poses, velocities and issued commands initialize the private
response. Physical limits and separate object tracking bounds apply at every
prediction step; final real-period verification and task audit remain separate.
No whole-task rollout selects structures here.
"""
import copy,threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np,mujoco,clarabel
from build.response_contact_control import ResponseContactControl
from build.object_tracking_domain import ObjectTrackingDomain,errors
from build.cached_response import CachedResponse
from build.native_contact_rollout import ContactRollout
from build.contact_step_control import pad_gap_jacobians

class ParallelSeparation:
    def __init__(self,separation,workers=8):
        self.separation=separation;self.thread=threading.local();self.pool=ThreadPoolExecutor(max_workers=workers)
    def one(self,item):
        if not hasattr(self.thread,'query'):
            local=copy.copy(self.separation);local.query=copy.copy(self.separation.query)
            local.query.data=mujoco.MjData(local.query.model);self.thread.query=local
        return self.thread.query.at(*item)
    def __call__(self,qs,vs):return self.pool.map(self.one,zip(qs,vs))
    def close(self):self.pool.shutdown()


class ContactCost(ObjectTrackingDomain):
    def __init__(self,parent,q,v):
        self.parent=parent;self.g=parent.optimizer;self.m=self.g.m
        target=parent.target.copy()
        mujoco.mj_integratePos(self.m,target,parent.target_velocity,-parent.horizon)
        super().__init__(self.m,q,v,target,parent.target_velocity)

    def residual(self,r,command,anchor):
        g=self.g;d=g.query;d.qpos[:]=r['q'];d.qvel[:]=r['v'];mujoco.mj_fwdPosition(self.m,d)
        gaps,J=pad_gap_jacobians(self.m,d,g.links,g.groups)
        load=.5-r['loads'][-1,:,0]
        # Only a deficient contact needs geometric restoration. Healthy
        # compression is not forced to a particular nominal gap.
        gap=np.where(load>0.,7500.*gaps+.5,0.)
        r['hold_gap_jacobian']=J;r['hold_gap_active']=gap>0;r['hold_load_active']=load>0
        return np.r_[np.maximum(gap,0.),np.maximum(load,0.),.001*(command-anchor)/g.input_scale]

    def jacobian(self,r,D,scale):
        gap=7500.*r['hold_gap_jacobian']@D[-1,:self.m.nv]
        load=-r['load_derivatives'][-1,:,0]*scale
        return np.vstack([gap*r['hold_gap_active'][:,None],load*r['hold_load_active'][:,None],.001*np.eye(self.m.nu)])

class HeldContactTracking:
    horizon = .02
    correction_period = .01

    def __init__(self, feedback):
        self.feedback = feedback
        self.model = copy.copy(feedback.m)
        # Prediction accuracy is independently checked against 0.125 ms execution.
        self.model.opt.timestep = .0005
        self.steps = round(self.horizon / self.model.opt.timestep)
        self.runner = ContactRollout(self.model, feedback.local.links)
        self.groups = [tuple(np.flatnonzero(self.runner.groups == i))
                       for i in range(len(feedback.local.links))]
        self.gradient = None
        self.calls = self.derivatives = 0
        self.correction = np.zeros(self.model.nu)
        query = mujoco.MjData(self.model)
        query.qpos[:] = feedback.nominal
        mujoco.mj_fwdPosition(self.model, query)
        self.air_gaps = pad_gap_jacobians(self.model, query, feedback.local.links, self.groups)[0]
        self.transfer_input = None

    def update(self, q, v, command, reference, velocity, input_reference):
        fb = self.feedback
        sample = self.calls % 4 == 0  # Existing contact loop is 2.5 ms.
        self.calls += 1
        if not sample:
            proposed = command + self.correction
            local = fb.local
            offset = local.kp*q[local.qi] + local.kd*v[local.vi]
            low = np.r_[(offset-local.effort)/local.gain, local.rotor_min]
            high = np.r_[(offset+local.effort)/local.gain, local.rotor_max]
            limited = self.model.actuator_ctrllimited.astype(bool)
            low[limited] = np.maximum(low[limited], self.model.actuator_ctrlrange[limited, 0])
            high[limited] = np.minimum(high[limited], self.model.actuator_ctrlrange[limited, 1])
            proposed = np.clip(proposed, low, high)
            accepted, _ = fb.contact_response.check(proposed, np.zeros(fb.m.nv))
            return proposed if accepted else command
        self.reference = self.air_gaps.copy()
        if input_reference is not None:
            if self.transfer_input is None:
                self.transfer_input = input_reference.copy()
                query = mujoco.MjData(self.model)
                query.qpos[:] = q
                mujoco.mj_fwdPosition(self.model, query)
                self.transfer_gaps = pad_gap_jacobians(self.model, query, fb.local.links, self.groups)[0]
            change = fb.u-self.transfer_input
            denominator = float(change @ change)
            ratio = 1. if denominator < 1e-20 else float(np.clip(
                (input_reference-self.transfer_input) @ change / denominator, 0., 1.))
            self.reference = (1-ratio)*self.transfer_gaps + ratio*self.air_gaps
        self.target = reference.copy()
        self.target_velocity = np.zeros(self.model.nv) if velocity is None else velocity.copy()
        mujoco.mj_integratePos(self.model, self.target, self.target_velocity, self.horizon)
        result = self.correct(q, v, command)
        self.correction = result-command
        return result

    def correct(self,q,v,command):
        if not hasattr(self,'optimizer'):
            # Keep the configured prediction model intact. Re-running the
            # PlantModel initializer can restore diagnostic CCD defaults.
            plant=copy.copy(self.feedback.local.plant)
            plant.mj=self.model
            self.optimizer=ResponseContactControl(plant,self.feedback.local.links,
                period=self.horizon,prediction_steps=self.steps,iterations=2,objective='contact_hold',
                clearance_model=self.feedback.clearance_model,fast_solve=True,adaptive_damping=True)
            self.optimizer.predictor=CachedResponse(plant,self.feedback.local.links,
                period=self.horizon,steps=self.steps,workers=16,
                fine_prefix_period=.0025,fine_timestep=self.feedback.m.opt.timestep)
            self.optimizer.feasibility_tolerance=1e-8
            self.parallel_separation=ParallelSeparation(self.optimizer.separation)
            self.optimizer.separation.batch=self.parallel_separation
            self.failures=0;self.accepted=0
        self.last_anchor=command.copy()
        g=self.optimizer;g.hold_cost=ContactCost(self,q,v);g.previous=command.copy()
        try:
            result=g.update(q,v,q,input_reference=command,normal_targets=-self.reference)
            self.accepted+=1
            self.feedback.info.update(constrained_lookahead_accepted=self.accepted,constrained_lookahead_unresolved=self.failures)
            return result
        except ValueError as error:
            self.failures+=1
            self.feedback.info.update(constrained_lookahead_accepted=self.accepted,constrained_lookahead_unresolved=self.failures)
            # A 20 ms constant-input failure does not establish physical
            # impossibility. The unchanged final 2.5 ms guard still validates.
            if self.failures<=3:print('lookahead unresolved',str(error)[:250],flush=True)
            return command
    def close(self):
        if hasattr(self,'optimizer'):self.optimizer.predictor.close();self.parallel_separation.close()
