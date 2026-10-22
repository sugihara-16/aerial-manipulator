"""Unload the support through the same finite contact and real-drive model."""
import numpy as np,mujoco
from build.contact_step_control import ContactStepControl
from build.check_local_pickup import smooth
from build.full_state_feedback import observed_tangent_speed
from build.separation_rate_constraints import preload_in_physical_domain


class FiniteLoadTransfer:
    def __init__(self,plant,clearance_model,links,ground,nominal,normal_targets):
        self.m=plant.mj;self.links=links;self.ground=ground.copy();self.nominal=nominal.copy()
        self.targets=normal_targets.copy();self.started=None;self.initial_support=None;self.ready_time=0.
        self.control=ContactStepControl(plant,links,regularized_friction=True,finite_support=True,
            distance_closing=True,condensed_solve=True,clearance_model=clearance_model)
        self.control.minimum_normal_load=.5
        self.free=np.flatnonzero(self.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        self.supports=[self.m.geom(n).id for n in ['support','destination_support','floor']]
        self.obj=self.m.geom('object').id;self.ready=False;self.info={}

    def update(self,q,v,physical_time):
        if self.started is None:self.started=float(physical_time)
        t=float(physical_time)-self.started
        if t>=8.:raise ValueError('finite_load_transfer_not_completed')
        if self.initial_support is not None:
            self.control.support_load_upper=max(0.,1-smooth(t/3.))*self.initial_support
        reference=self.ground.copy()
        for j in self.free:reference[self.m.jnt_qposadr[j]+2]+=.15*smooth((t-3.)/4.)
        command=self.control.update(q,v,reference,input_reference=self.nominal,normal_targets=self.targets)
        load=self.control.diagnostics['predicted_support_normal_load']
        if self.initial_support is None:self.initial_support=max(0.,load)
        m,d=self.m,self.control.d;distances=[]
        for geom in self.supports:
            segment=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,geom,self.obj,1.,segment);rate=0.
            if 0.<gap<1.:
                normal=segment[3:]-segment[:3];normal/=np.linalg.norm(normal)
                jacobian=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jacobian,None,segment[3:],int(m.geom_bodyid[self.obj]))
                rate=float(normal@jacobian@v)
            distances.append((gap,rate))
        gap,rate=min(distances)
        admissible=gap>0. and rate>=0. and observed_tangent_speed(m,d,self.links)<1e-4 and preload_in_physical_domain(m,d,self.links)
        self.ready_time=self.ready_time+.01 if admissible else 0.;self.ready=self.ready_time>=.3
        self.info=dict(elapsed_s=t,initial_support_n=self.initial_support,support_cap_n=self.control.support_load_upper,
                       estimated_support_n=load,support_gap_m=gap,support_rate_m_s=rate,ready_time_s=self.ready_time)
        return command
