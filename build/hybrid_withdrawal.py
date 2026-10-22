"""Empty-flight balance followed by the common certified withdrawal planner.

When the present rotor posture cannot support the unloaded robot, use the
continuous rotor-trim transition. This branch is selected by static mechanics,
never by object identity or the result of a simulated task.
"""
import time
import numpy as np
from build.algebraic_release import SupportedWithdrawal,coupled
from build.unloaded_hover import command as gravity_command
from build.planned_withdrawal import PlannedWithdrawal
from amsrr_det.release import plan_opening
from amsrr_det.free_flight_trim import scheduled_support
from amsrr_det.retreat import rest_to_rest_durations
from amsrr_det.numerics import quintic_reference


class HybridWithdrawal(SupportedWithdrawal):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.transition=None

    def prepare(self,q,v,previous):
        began=time.perf_counter();p=self.plant;plan=self.plan
        self.query.qpos[:]=q;self.query.qvel[:]=v;qp,_=p.observation(self.query)
        obstacles=[(self.m.geom(name).pos.copy(),self.m.geom(name).size.copy())for name in ['support','destination_support']]
        path=plan_opening(p.pin,p.meta,plan.shape,qp,qp,plan.surfaces,plan.points,plan.normals,obstacles,plan.destination)
        self.path_data=path;self.durations=rest_to_rest_durations(p.pin,p.meta,path['path'],path['distances'],normal_speed=path['segment_normal_speeds'])
        self.ends=np.r_[0.,np.cumsum(self.durations)]
        self.flight=coupled.CoupledFlightController(p.pin,p.meta,qp,track_vectoring=True,planned_input_reference=True)
        self.flight.command=np.where(self.local.dock,previous[:self.local.nj]/self.local.kp,previous[:self.local.nj]);self.flight.force=previous[self.local.nj:].copy()
        self.entry=self.elapsed;self.ready=True;self.planning_s=time.perf_counter()-began
        self.info=dict(planning_s=self.planning_s,duration_s=float(self.ends[-1]),withdrawal=path['withdrawal'],clearance=path['clearance'],repairs=path['clearance_repairs'])

    def update(self,q,v,previous):
        if self.transition is not None:
            u=self.transition.update(q,v,previous);self.ready=self.transition.ready;self.complete=self.transition.complete;self.info=self.transition.info
            return u
        if not self.ready:
            if self.elapsed<.3-1e-9:
                try:u=gravity_command(self.local,q,previous)[0]
                except ValueError as error:
                    if str(error)!='no unloaded gravity allocation inside motor bounds':raise
                    self.transition=PlannedWithdrawal(self.plan,self.plant,q)
                    return self.update(q,v,previous)
                self.elapsed+=.01;return u
            self.prepare(q,v,previous)
        elapsed=self.elapsed-self.entry;index=min(len(self.durations)-1,int(np.searchsorted(self.ends,elapsed,side='right')-1));path=self.path_data
        target,tv,ta,s=quintic_reference(self.plant.pin,self.plant.meta,path['path'][index],path['path'][index+1],elapsed-self.ends[index],self.durations[index])
        bias,thrust=scheduled_support(path['schedules'][index],s)
        self.query.qpos[:]=q;self.query.qvel[:]=v;qp,vp=self.plant.observation(self.query)
        cmd,force=self.flight.update(qp,vp,target,target_velocity=tv,target_acceleration=ta,joint_bias=bias,thrust_reference=thrust,gimbal_reference=target[self.flight.qi[self.flight.ai]])
        if self.flight.allocation_failure:raise ValueError(self.flight.allocation_failure)
        self.complete=bool(elapsed>=self.ends[-1]);self.elapsed+=.01
        return np.r_[np.where(self.local.dock,self.local.kp*cmd,cmd),force]
