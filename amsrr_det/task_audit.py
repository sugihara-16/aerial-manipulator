"""Privileged simulation evidence. Never provided to the task controller."""
import numpy as np
import mujoco
from .clearance import PART_CLEARANCE, ENVIRONMENT_CLEARANCE, MotionBoundedClearance


class TaskAudit:
    def __init__(self,plant,selected_links,object_size,support_top,shape=None,clearance_model=None, *, support_geometry=False):
        self.plant=plant;self.links=selected_links;self.size=np.asarray(object_size)
        if plant.mj.opt.cone!=mujoco.mjtCone.mjCONE_ELLIPTIC:
            raise ValueError('audit requires elliptic contact forces')
        self.shape=shape;self.support_top=support_top;self.max_force=0.;self.max_moment=0.
        self.max_penetration=0.;self.max_joint_speed=0.;self.max_effort_ratio=0.
        self.max_joint_limit_violation=0.;self.max_clearance=0.;self.other_contacts=set()
        self.min_clearance_carried=float('inf');self.max_slip=0.;self.anchors={}
        self.material_anchors={};self.max_material_displacement=0.
        self.clearance_violations=set();self.minimum_part_gap=PART_CLEARANCE;self.minimum_environment_gap=ENVIRONMENT_CLEARANCE
        self.proximity=MotionBoundedClearance(plant.mj if clearance_model is None else clearance_model,selected_links,omit_permitted_support=not support_geometry)
        self.proximity_model=self.proximity.model;self.proximity_data=self.proximity.data
        self.names=[mujoco.mj_id2name(plant.mj,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(plant.mj.ngeom)]
        self.object_body=mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_BODY,'object')
        self.link_bodies={n:mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_BODY,n) for n in selected_links}
        self.ai=np.array([mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_ACTUATOR,a['name']) for a in plant.meta['actuators']])
        self.jids=np.array([mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_JOINT,a['name']) for a in plant.meta['actuators']])
        self.qadr=plant.mj.jnt_qposadr[self.jids];self.vadr=plant.mj.jnt_dofadr[self.jids]
        self.limits=np.array([[a['lower'],a['upper']] for a in plant.meta['actuators']])
        self.effort=np.array([a['effort'] for a in plant.meta['actuators']])
        self.saturated_time=np.zeros(len(self.ai));self.max_raw_effort_ratio=0.;self.max_servo_error=0.
        self.groups=np.array([next((j for j,n in enumerate(self.links) if name.startswith(n+'/')),-1) for name in self.names])
        self.object_geom=mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_GEOM,'object')
        self.allowed_support=np.array([name in ['support','destination_support','floor'] for name in self.names])
        support_types=plant.mj.geom_type[self.allowed_support]
        self.support_representation=('native_unsplit' if np.all(np.isin(support_types,
            [mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_PLANE])) else 'other')
        self.gap_time=np.zeros(len(self.links));self.longest_gap=0.;self.last_contact_time=None;self.time=0.;self.last_normal_loads=np.zeros(len(self.links))

    def update(self,data, *, carried=False,grasp_contact_allowed=True):
        m=self.plant.mj;body=self.object_body;self.time=float(data.time)
        pos=data.xpos[body];R=data.xmat[body].reshape(3,3)
        if self.shape is None:
            clearance=float(pos[2]-abs(R[2])@(self.size/2)-self.support_top)
        else:
            quat=np.zeros(4);mujoco.mju_mat2Quat(quat,R.ravel())
            clearance=self.shape.lowest_z(np.r_[pos,quat[[1,2,3,0]]])-self.support_top
        self.max_clearance=max(self.max_clearance,clearance)
        if carried:self.min_clearance_carried=min(self.min_clearance_carried,clearance)
        self.max_joint_speed=max(self.max_joint_speed,float(np.max(abs(data.qvel[self.vadr]))))
        self.max_effort_ratio=max(self.max_effort_ratio,float(np.max(abs(data.actuator_force[self.ai])/self.effort)))
        raw=(m.actuator_gainprm[self.ai,0]*data.ctrl[self.ai]+m.actuator_biasprm[self.ai,1]*data.actuator_length[self.ai]
             +m.actuator_biasprm[self.ai,2]*data.actuator_velocity[self.ai])
        self.max_raw_effort_ratio=max(self.max_raw_effort_ratio,float(np.max(abs(raw)/self.effort)))
        self.saturated_time+=(abs(raw)>self.effort)*m.opt.timestep
        self.max_servo_error=max(self.max_servo_error,float(abs(data.ctrl[self.ai]-data.actuator_length[self.ai]).max()))
        q=data.qpos[self.qadr]
        self.max_joint_limit_violation=max(self.max_joint_limit_violation,float(np.max(np.maximum(self.limits[:,0]-q,q-self.limits[:,1]))))
        contact=data.contact
        g1,g2=np.asarray(contact.geom1),np.asarray(contact.geom2)
        gap=np.asarray(contact.dist)
        if len(gap):self.max_penetration=max(self.max_penetration,float(-gap.min()))
        object_first=g1==self.object_geom;object_second=g2==self.object_geom
        other=np.where(object_first,g2,g1);group=self.groups[other]
        unexpected=((~object_first & ~object_second) |
                    ((object_first|object_second)&((group<0)|~np.bool_(grasp_contact_allowed))&~self.allowed_support[other])) & (gap < -1e-5)
        for i in np.flatnonzero(unexpected):
            self.other_contacts.add(tuple(sorted([self.names[g1[i]],self.names[g2[i]]])))
        violations,part,environment=self.proximity.check(data.qpos,grasp_contact_allowed)
        self.minimum_part_gap=min(self.minimum_part_gap,part)
        self.minimum_environment_gap=min(self.minimum_environment_gap,environment)
        self.clearance_violations.update(violations)
        selected=(object_first|object_second)&(group>=0)&(np.asarray(contact.efc_address)>=0)
        indices=np.flatnonzero(selected);groups=group[indices]
        rows=np.asarray(contact.efc_address)[indices]
        # Elliptic condim=3 contact forces are the three constraint multipliers.
        # This is the same representation returned by mj_contactForce.
        if len(indices) and np.any(np.asarray(contact.dim)[indices]!=3):raise ValueError('audit requires 3D point contacts')
        local=data.efc_force[rows[:,None]+np.arange(3)]
        frames=np.asarray(contact.frame).reshape(-1,3,3)[indices]
        force=np.einsum('nij,ni->nj',frames,local)*np.where(object_first[indices],1.,-1.)[:,None]
        points=np.asarray(contact.pos)[indices]
        origins=np.array([data.xpos[self.link_bodies[self.links[g]]] for g in groups]).reshape(-1,3)
        moment=np.cross(points-origins,force)
        total_force=np.zeros((len(self.links),3));total_moment=total_force.copy()
        np.add.at(total_force,groups,force);np.add.at(total_moment,groups,moment)
        self.max_force=max(self.max_force,float(np.linalg.norm(total_force,axis=1).max(initial=0)))
        self.max_moment=max(self.max_moment,float(np.linalg.norm(total_moment,axis=1).max(initial=0)))
        weights=np.maximum(local[:,0],0.);totals=np.zeros(len(self.links));centers=total_force*0
        np.add.at(totals,groups,weights)
        self.last_normal_loads=totals.copy()
        if np.any(totals>.05):self.last_contact_time=self.time
        np.add.at(centers,groups,weights[:,None]*((points-pos)@R))
        if carried:
            self.gap_time=np.where(totals>.5,0.,self.gap_time+m.opt.timestep)
            self.longest_gap=max(self.longest_gap,float(self.gap_time.max(initial=0)))
            for i,name in enumerate(self.links):
                if totals[i]<=.5:continue
                center=centers[i]/totals[i]
                if name not in self.anchors:self.anchors[name]=center
                self.max_slip=max(self.max_slip,float(np.linalg.norm(center-self.anchors[name])))
                # A pressure-weighted center can move across an unchanged flat
                # contact patch. Track the SAME gripper material point relative
                # to the object instead of calling that load redistribution slip.
                bid=self.link_bodies[name];rotation=data.xmat[bid].reshape(3,3);origin=data.xpos[bid]
                if name not in self.material_anchors:
                    self.material_anchors[name]=((R@center+pos-origin)@rotation,center.copy())
                local,initial=self.material_anchors[name]
                current=(rotation@local+origin-pos)@R
                self.max_material_displacement=max(self.max_material_displacement,float(np.linalg.norm(current-initial)))

    def result(self):
        return dict(engine='MuJoCo',contact_force_components='normal_and_friction',
                    clearance_detection='refitted_body_bounds',support_representation=self.support_representation,
                    clearance_queries=self.proximity.queries,clearance_motion_bound_reuses=self.proximity.reused,
                    peak_contact_force_n=self.max_force,peak_contact_moment_nm=self.max_moment,
                    peak_penetration_m=self.max_penetration,peak_joint_speed_rad_s=self.max_joint_speed,
                    peak_effort_ratio=self.max_effort_ratio,joint_limit_violation_rad=self.max_joint_limit_violation,
                    peak_unsaturated_servo_effort_ratio=self.max_raw_effort_ratio,maximum_servo_reference_error_rad=self.max_servo_error,
                    actuator_saturation_time_s={a['name']:float(t) for a,t in zip(self.plant.meta['actuators'],self.saturated_time) if t>0},
                    maximum_lift_clearance_m=self.max_clearance,
                    minimum_transport_clearance_m=self.min_clearance_carried if np.isfinite(self.min_clearance_carried) else None,
                    time_since_gripper_contact_s=None if self.last_contact_time is None else self.time-self.last_contact_time,
                    minimum_part_gap_capped_m=self.minimum_part_gap,minimum_environment_gap_capped_m=self.minimum_environment_gap,
                    clearance_violations=sorted(self.clearance_violations),
                    contact_center_displacement_m=self.max_slip,
                    contact_material_displacement_m=self.max_material_displacement,
                    longest_carried_contact_gap_s=self.longest_gap,unexpected_contacts=sorted(self.other_contacts))
