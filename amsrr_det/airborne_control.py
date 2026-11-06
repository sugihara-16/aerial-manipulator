"""Existing articulated flight allocation with cooperative joint-load tracking."""
import numpy as np
import mujoco
import pinocchio as pin
from .flight import FlightController


class AirborneControl:
    def __init__(self,plant):
        self.plant=plant;self.m=plant.mj;self.d=mujoco.MjData(self.m)
        self.controller=FlightController(plant.pin,plant.meta,pin.neutral(plant.pin),guard_vectoring=True,cooperative_joints=True)
        self.base=np.array([self.m.jnt_dofadr[self.m.joint(plant.pin.names[b['joint']]).id] for b in plant.meta['bases']])
        self.last={}

    def seed(self,q,v,command):
        n=len(self.plant.meta['actuators']);self.controller.command=command[:n].copy();self.controller.force=command[n:].copy();self.controller.trim_effort=None

    def track_applied(self,command):
        n=len(self.plant.meta["actuators"])
        self.controller.command=command[:n].copy();self.controller.force=command[n:].copy()

    def update(self,q,v,reference,velocity,acceleration,now,nominal_input=None,feedback_bandwidth=1.,acceleration_requests=None):
        p=self.plant;self.d.qpos[:]=q;self.d.qvel[:]=v;pq,pv=p.observation(self.d)
        self.d.qpos[:]=reference;self.d.qvel[:]=0.;target,_=p.observation(self.d)
        _,_,pvi,mvi,bases=p.coordinate_maps
        tv=np.zeros(p.pin.nv);ta=tv.copy();tv[pvi]=velocity[mvi];ta[pvi]=acceleration[mvi]
        for pos,vel,source in bases:
            R=pin.Quaternion(pq[pos+3:pos+7]).matrix()
            tv[vel+3:vel+6]=R@tv[vel+3:vel+6];ta[vel+3:vel+6]=R@ta[vel+3:vel+6]
        cmd,force=self.controller.update(pq,pv,target,target_velocity=tv,target_acceleration=ta,nominal_input=nominal_input,feedback_bandwidth=feedback_bandwidth,acceleration_requests=acceleration_requests)
        self.last=dict(failure=self.controller.allocation_failure,solver_status=[d['solver_status']for d in self.controller.allocation_diagnostics])
        return np.r_[cmd,force]
