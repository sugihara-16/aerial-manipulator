"""Reuse geometric withdrawal planning with rotor-direction equilibrium fits.

The supported object is unloaded continuously before a certified opening and
translation path. Planning uses geometry and static mechanics, not rollouts.
"""
import numpy as np,mujoco
from amsrr_det.grasp_control import GraspReferences
from amsrr_det.release import SupportedRelease
from build.algebraic_release import coupled


class PlannedWithdrawal:
    def __init__(self,plan,plant,q):
        self.plan,self.plant=plan,plant;self.query=mujoco.MjData(plant.mj)
        self.reference=None;self.ready=False;self.complete=False;self.info={}

    def update(self,q,v,previous):
        self.query.qpos[:]=q;self.query.qvel[:]=v;qp,vp=self.plant.observation(self.query)
        if self.reference is None:
            p=self.plant;plan=self.plan
            self.dock=np.array([a['role']=='dock'for a in p.meta['actuators']]);self.kp=np.array([a['kp']for a in p.meta['actuators']]);self.nj=len(self.kp)
            self.flight=coupled.CoupledFlightController(p.pin,p.meta,qp,track_vectoring=True,planned_input_reference=True)
            self.flight.command=np.where(self.dock,previous[:self.nj]/self.kp,previous[:self.nj]);self.flight.force=previous[self.nj:].copy()
            grasp=GraspReferences(p.pin,p.meta,qp,qp,plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape)
            self.reference=SupportedRelease(plan,grasp,self.flight,qp,rest_to_rest=True)
        reference=self.reference
        if reference.unloaded_reference_settled:
            target,tv,_,bias,thrust=reference.open(qp,vp);self.ready=reference.retreat is not None
        else:target,tv,_,bias,thrust=reference.unload()
        cmd,forces=self.flight.update(qp,vp,target,target_velocity=tv,joint_bias=bias,thrust_reference=thrust,
            gimbal_reference=target[self.flight.qi[self.flight.ai]])
        if self.flight.allocation_failure:raise ValueError(self.flight.allocation_failure)
        self.complete=bool(reference.opening_reference_complete)
        self.info=dict(planning_s=sum(reference.planning_times),unload_duration_s=reference.duration,
            opening=getattr(reference,'opening_evidence',None))
        return np.r_[np.where(self.dock,self.kp*cmd,cmd),forces]
