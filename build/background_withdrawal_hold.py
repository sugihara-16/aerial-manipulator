"""Diagnostic: hold the planning pose with the existing empty-flight control.

The geometric plan, actual command limits and ordinary unload are unchanged.
This adds pose/velocity feedback during planning, where gravity balance alone
otherwise lets the observed planning configuration drift.
"""
import numpy as np
from build.background_withdrawal import BackgroundHybridWithdrawal
from build.unloaded_hover import command as gravity_command
from build.algebraic_release import coupled


class HeldBackgroundWithdrawal(BackgroundHybridWithdrawal):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.hold_flight=None

    def update(self,q,v,previous):
        if self.transition is None and not self.ready and self.elapsed>=.3-1e-9 and self.hold_flight is None:
            self.hold_input=gravity_command(self.local,q,previous)[0]
            self.query.qpos[:]=q;self.query.qvel[:]=v
            self.hold_target,_=self.plant.observation(self.query);self.hold_target=self.hold_target.copy()
            self.hold_bias=self.local.gain*self.hold_input[:self.local.nj]-self.local.kp*q[self.local.qi]
            self.hold_flight=coupled.CoupledFlightController(self.plant.pin,self.plant.meta,self.hold_target,
                track_vectoring=True,planned_input_reference=True)
            self.hold_flight.command=np.where(self.local.dock,previous[:self.local.nj]/self.local.kp,previous[:self.local.nj])
            self.hold_flight.force=previous[self.local.nj:].copy()
        command=super().update(q,v,previous)
        if self.job is not None and self.transition is None:
            self.query.qpos[:]=q;self.query.qvel[:]=v;qp,vp=self.plant.observation(self.query)
            cmd,force=self.hold_flight.update(qp,vp,self.hold_target,
                joint_bias=self.hold_bias,thrust_reference=self.hold_input[self.local.nj:],
                gimbal_reference=self.hold_target[self.hold_flight.qi[self.hold_flight.ai]])
            if self.hold_flight.allocation_failure:raise ValueError(self.hold_flight.allocation_failure)
            command=np.r_[np.where(self.local.dock,self.local.kp*cmd,cmd),force]
        return command
