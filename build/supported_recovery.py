"""Local grasp recovery near the source support, including brief separation.

Uses observed poses/velocities and the existing finite one-period acquisition
model, never evaluation contact forces. Does not reset physical/audit history.
Only geometrically present support contacts enter the finite contact solve.
The entry guard and first feasible command are local checks, not a certificate
that recapture or the subsequent lift will succeed.
"""
from functools import partial
import numpy as np
import mujoco
from build.contact_step_control import ContactStepControl, pad_gap_jacobians
from build.generated_grasp_stage import align_reference_to_object, generate
from build.full_state_feedback import observed_tangent_speed
from build.background_plan import BackgroundPlan
from build.separation_rate_constraints import preload_in_physical_domain,preload_domain_rates


class SupportedRecovery:
    def __init__(self, plant, clearance_model, plan, links, goal, targets):
        self.plant,self.plan,self.links=plant,plan,links
        self.m=plant.mj;self.query=mujoco.MjData(self.m)
        self.goal=goal.copy();self.targets=targets.copy()
        self.control=ContactStepControl(plant,links,period=.005,
            regularized_friction=True,finite_support=True,distance_closing=True,
            condensed_solve=True,clearance_model=clearance_model,inertial_updates=True)
        self.clearance_model=clearance_model
        self.control.separation=RetainSupport(self.control.separation,self.m,self.query,self.control.h,self.links)
        self.control.object_velocity_bounds=(-np.r_[[.02]*3,[.1]*3],np.r_[[.02]*3,[.1]*3])
        self.control.object_velocity_norm_limits=(.02,.1)
        from build.held_input_guard import HeldInputGuard
        self.held_guard=HeldInputGuard(plant,links,clearance_model,period=self.control.h,
            object_velocity_norm_limits=self.control.object_velocity_norm_limits)
        self.ready_time=0.;self.contact_ready_time=0.;self.job=None;self.stage=None;self.pending=None;self.info={}
        self.anchor=None

    def supported(self,q,v):
        m,d=self.m,self.query;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_fwdPosition(m,d)
        obj=m.geom('object').id;oi=m.jnt_qposadr[m.joint('object').id];ov=m.jnt_dofadr[m.joint('object').id]
        # Same 1e-7 numerical residual tolerance as the physical input checks.
        if np.linalg.norm(v[ov:ov+3])>.02+1e-7 or np.linalg.norm(v[ov+3:ov+6])>.1+1e-7:
            raise ValueError('supported recovery object is not settled')
        # Only the source support is eligible, not an arbitrary floor contact.
        support=m.geom('support').id
        contacts=[c for c in d.contact if set(map(int,c.geom))=={obj,support} and c.dist<=0.]
        gap=mujoco.mj_geomDistance(m,d,obj,support,.03,np.zeros(6))
        error=q[oi:oi+3]-self.plan.source[:3]
        if gap>=.03 or np.linalg.norm(error)>=.03:
            raise ValueError('recovery left the source neighborhood')
        lower=np.maximum(-.02,(-.03/np.sqrt(3)-error)/self.control.h)
        upper=np.minimum(.02,(.03/np.sqrt(3)-error)/self.control.h)
        self.control.object_velocity_bounds=(np.r_[lower,[-.1]*3],np.r_[upper,[.1]*3])
        self.info.update(source_supported=bool(contacts),source_gap_m=float(gap))
        R=d.geom_xmat[support].reshape(3,3);relative=R.T@(q[oi:oi+3]-d.geom_xpos[support])
        # Keep the COM inside the actual support footprint; the coupled force
        # solve below additionally checks the remaining grasp and support.
        if np.any(abs(relative[:2])>=m.geom_size[support,:2]):
            raise ValueError('supported recovery COM is outside the source support')
        gaps,_=pad_gap_jacobians(m,d,self.links,self.control.groups)
        if np.all(gaps>0.):raise ValueError('supported recovery requires remaining grasp contact')
        self.info.update(object_speed_m_s=float(np.linalg.norm(v[ov:ov+3])),
                         object_angular_speed_rad_s=float(np.linalg.norm(v[ov+3:ov+6])),
                         pad_gaps_m=gaps.tolist())
        return gaps

    def update(self,q,v,previous,simulation_time,material_travel):
        gaps=self.supported(q,v)
        if self.anchor is None:self.anchor=previous.copy()
        self.control.previous_input=previous.copy()
        reference=align_reference_to_object(self.m,self.goal,q)
        command=self.control.update(q,v,reference,input_reference=self.anchor,normal_targets=self.targets)
        command=self.held_guard.update(q,v,command)
        settled=max(gaps)<=0. and observed_tangent_speed(self.m,self.query,self.links)<1e-4
        self.contact_ready_time=self.contact_ready_time+.005 if settled else 0.
        preload_ready=settled and preload_in_physical_domain(self.m,self.query,self.links)
        self.info.update(contact_settled=bool(settled),preload_ready=bool(preload_ready),
            tangent_speed_m_s=float(observed_tangent_speed(self.m,self.query,self.links)))
        self.ready_time=self.ready_time+.005 if preload_ready else 0.
        # Design the new preload from settled geometry. Requiring that new
        # preload before its design would strand over-compressed captures.
        if self.contact_ready_time>=.3 and self.job is None and self.pending is None:
            operation=partial(generate,momentum_feedback=True,finite_slip_preload=True,
                bounded_allocation=True,contact_coupled=True,minimize_input=True,
                governed_motion=True,fixed_observation=True,nearby_observation=True,
                material_travel_used=float(material_travel))
            self.job=BackgroundPlan(simulation_time,operation,self.plant,self.clearance_model,q.copy(),self.links,self.plan)
        if self.job is not None and self.job.ready(simulation_time):
            self.info['observation_s']=self.job.started_sim
            candidate,wait=self.job.result(simulation_time);self.job=None
            self.info['planning']=wait
            if not candidate['available']:
                # A failed local posture solve does not invalidate the
                # feasible recovery command. Settle again before retrying;
                # the task's overall execution deadline remains unchanged.
                self.info['unavailable_refinement']=candidate['refinement']
                self.info['refinement_retries']=self.info.get('refinement_retries',0)+1
                self.contact_ready_time=0.
                return command
            self.pending=candidate;self.goal=candidate['goal'];self.targets=candidate['normal_targets']
            self.anchor=candidate['feedback'].u.copy();self.ready_time=0.
        if self.pending is not None and self.ready_time>=.3:
            from build.generated_grasp_stage import entry_error
            if entry_error(self.pending['feedback'],q,v,self.pending['goal'])<=1.:
                self.stage=self.pending
        return command

    def close(self):
        if self.job is not None:self.job.close();self.job=None
        self.held_guard.close()


class RetainSupport:
    """Retain an existing source contact; never invent one across a gap."""
    def __init__(self,clearance,model,query,period,links):
        self.clearance,self.m,self.d,self.h=clearance,model,query,period
        self.links=links
    def at(self,q,v):
        rows,bounds=self.clearance.at(q,v)
        m,d=self.m,self.d;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_fwdPosition(m,d)
        pair={m.geom('object').id,m.geom('support').id}
        contacts=[c for c in d.contact if set(map(int,c.geom))==pair and c.dist<=0.]
        if not contacts:
            extra,limits=preload_domain_rates(m,d,self.links,self.h,maintain_floor=True)
            return np.vstack([rows,extra]),np.r_[bounds,limits]
        c=min(contacts,key=lambda c:c.dist);ga,gb=map(int,c.geom)
        ja=np.zeros((3,m.nv));jb=ja.copy()
        mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[ga]))
        mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[gb]))
        J=c.frame[:3]@(jb-ja)
        rows=np.vstack([rows,-J]);bounds=np.r_[bounds,float(c.dist)/self.h]
        extra,limits=preload_domain_rates(m,d,self.links,self.h,maintain_floor=True)
        return np.vstack([rows,extra]),np.r_[bounds,limits]
