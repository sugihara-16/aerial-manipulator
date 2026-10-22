"""Free-flight approach to a geometrically separated, stationary handoff.

The original grasp endpoint may already compress a pad. Stop the approach
before that first contact, recompute empty-flight thrust/bias for the actual
waypoints, and let the contact controller perform the remaining acquisition.
"""
import numpy as np,mujoco,pinocchio as pin
from build.contact_step_control import pad_gap_jacobians
from amsrr_det.free_flight_trim import free_flight_trim,support_schedule,scheduled_support
from amsrr_det.release import check_opening_clearance
from amsrr_det.numerics import quintic_reference


class StagedApproach:
    def prepared(self):
        """Transfer this newly calculated path, without transferring simulators."""
        return dict(first=self.first,last=self.last,schedule=self.schedule,
                    duration=self.duration,info=self.info)

    @classmethod
    def from_prepared(cls,plan,plant,links,state):
        path=cls.__new__(cls)
        path.plan,path.plant,path.links=plan,plant,links
        path.d=mujoco.MjData(plant.mj)
        path.first,path.last=state['first'].copy(),state['last'].copy()
        path.schedule,path.duration,path.info=state['schedule'],state['duration'],state['info']
        return path

    def __init__(self,plan,plant,goal,links,standoff=.01,duration=4.):
        self.plan,self.plant=plan,plant;self.duration=duration;self.d=mujoco.MjData(plant.mj)
        self.d.qpos[:]=goal;target,_=plant.observation(self.d);first=plan.open_q.copy()
        def gaps(q):
            self.d.qpos[:]=plant.configuration(q);mujoco.mj_fwdPosition(plant.mj,self.d)
            return pad_gap_jacobians(plant.mj,self.d,links)[0]
        if min(gaps(first))<=standoff:raise ValueError('open posture has no acquisition standoff')
        lo,hi=0.,1.
        for _ in range(32):
            mid=(lo+hi)/2;q=pin.interpolate(plant.pin,first,target,mid)
            if min(gaps(q))>=standoff:lo=mid
            else:hi=mid
        last=pin.interpolate(plant.pin,first,target,lo)
        first,t0=free_flight_trim(plant.pin,plant.meta,first)
        last,t1=free_flight_trim(plant.pin,plant.meta,last)
        if t0['status']!='static_screen_pass'or t1['status']!='static_screen_pass':raise ValueError('free approach equilibrium unavailable')
        # Do not alter the already-spawned start posture to hide a trim change.
        if max(abs(pin.difference(plant.pin,plan.open_q,first)))>1e-8:raise ValueError('spawned open posture requires a trim transition')
        self.first,self.last=first,last
        self.schedule=support_schedule(plant.pin,plant.meta,first,last,t0,t1)
        obstacles=[(plant.mj.geom(name).pos.copy(),plant.mj.geom(name).size.copy())for name in ['support','destination_support']]
        certificate=check_opening_clearance(plant.meta,plan.shape,plan.surfaces,[first,last],obstacles,
                                          prepared_plant=plan.plant)
        self.info=dict(standoff_m=standoff,duration_s=duration,minimum_endpoint_gap_m=float(min(gaps(last))),clearance=certificate,schedule_nodes=len(self.schedule))
        self.links=links

    def ready(self,q,v,time):
        if time<self.duration:return False
        self.d.qpos[:]=q;mujoco.mj_fwdPosition(self.plant.mj,self.d)
        _,J=pad_gap_jacobians(self.plant.mj,self.d,self.links)
        # An uncontacted object can roll. Handoff waits for the approaching
        # robot to settle, not for an independent object to stop moving.
        robot_velocity=v.copy();oi=int(self.plant.mj.jnt_dofadr[self.plant.mj.joint('object').id]);robot_velocity[oi:oi+6]=0.
        return bool(max(abs(J@robot_velocity))<=.0002)

    def reference(self,time):
        q,v,a,r=quintic_reference(self.plant.pin,self.plant.meta,self.first,self.last,time,self.duration)
        bias,thrust=scheduled_support(self.schedule,r)
        return q,v,a,r,bias,thrust
