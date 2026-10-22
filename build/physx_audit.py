"""PhysX contact evidence and engine-independent geometry/joint checks.

Simulation contact data only enters this evaluator. Controllers receive q/v.
The separate MuJoCo model only performs distance/pose calculations, never steps.
"""
import numpy as np
import mujoco
from amsrr_det.task_audit import TaskAudit
from amsrr_det.clearance import PART_CLEARANCE, ENVIRONMENT_CLEARANCE


class PhysXAudit(TaskAudit):
    def __init__(self,plant,links,shape,support_top,sim,geometry):
        super().__init__(plant,links,shape.size,support_top,shape,support_geometry=True)
        self.sim=sim;self.geometry=geometry;self.contacts=[];self.contact_shapes=[]
        self.friction=[];self.friction_shapes=[];self.reported_friction=0;self.extracted_friction=0
        # The ordinary report contains normal impulses only. Friction acts at
        # separate anchors and must contribute to the pad force AND moment.
        self.subscription=sim.interface.subscribe_full_contact_report_events(self._contacts)
        self.max_nominal_geometry_penetration=0.
        self.max_independent_support_penetration=0.;self.reported_points=0;self.extracted_points=0
        self.support_layer_points={}
        self.max_internal_support_impulse=0.;self.internal_support_example=None
        self.max_internal_support_friction_impulse=0.
        self.qi=np.array([a['q'] for a in plant.meta['actuators']]);self.vi=np.array([a['v'] for a in plant.meta['actuators']])
        self.kp=np.array([a['kp'] for a in plant.meta['actuators']]);self.kd=np.array([a['kd'] for a in plant.meta['actuators']])

    def _contacts(self,headers,data,anchors=()):
        for header in headers:
            friction_count=int(getattr(header,'num_friction_anchors_data',0))
            if not header.num_contact_data and not friction_count:continue
            self.reported_points+=int(header.num_contact_data)
            ids=[int(getattr(header,key)) for key in ['collider0','collider1']]
            first,second=(self.geometry['lookup'][key] for key in ids)
            internal_layers=set()
            if self.object_geom in [first,second]:
                for key in ids:
                    layer=self.geometry.get('support_layer_ids',{}).get(key)
                    if layer:
                        name=self.names[layer['geom']]+'/'+layer['layer']
                        self.support_layer_points[name]=self.support_layer_points.get(name,0)+int(header.num_contact_data)
            for c in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]:
                self.extracted_points+=1
                position,impulse,normal=c.position,c.impulse,c.normal
                if self.object_geom in [first,second]:
                    # A lower layer's upward-facing surface and an upper
                    # layer's downward-facing surface are newly cut material.
                    # Their edge normals also have that inward z component;
                    # legitimate vertical outer sides and outer top/bottom do
                    # not. Record actual impulse, not speculative points alone.
                    for side,key in enumerate(ids):
                        layer=self.geometry.get('support_layer_ids',{}).get(key)
                        if not layer:continue
                        outward=np.array([normal.x,normal.y,normal.z])*(1. if side==1 else -1.)
                        local=layer['rotation'].T@outward
                        internal=(local[2]>1e-5 if layer['layer']=='layer_1' else local[2]<-1e-5)
                        if internal:internal_layers.add(key)
                        magnitude=float(np.linalg.norm([impulse.x,impulse.y,impulse.z])) if internal else 0.
                        if magnitude>self.max_internal_support_impulse:
                            self.max_internal_support_impulse=magnitude
                            self.internal_support_example=dict(layer=self.names[layer['geom']]+'/'+layer['layer'],outward_normal=outward.tolist(),point=[position.x,position.y,position.z],separation=float(c.separation),impulse_ns=magnitude)
                self.contacts.append((first,second,position.x,position.y,position.z,impulse.x,impulse.y,impulse.z,normal.x,normal.y,normal.z,c.separation))
                self.contact_shapes.append(tuple(ids))
            self.reported_friction+=friction_count
            if friction_count:
                offset=header.friction_anchors_offset
                for anchor in anchors[offset:offset+friction_count]:
                    point,impulse=anchor.position,anchor.impulse
                    if self.object_geom in [first,second]:
                        for key in ids:
                            layer=self.geometry.get('support_layer_ids',{}).get(key)
                            if not layer:continue
                            possible=key in internal_layers
                            # Friction anchors have no normal/face index. An
                            # anchor near the interior cut, strictly inside
                            # the slab footprint, cannot be certified as an
                            # outer-face load. Reject that ambiguity too.
                            if 'center' in layer and 'half_size' in layer:
                                local=layer['rotation'].T@(np.array([point.x,point.y,point.z])-layer['center'])
                                half=layer['half_size'];plane=half[2]-.004
                                possible |= bool(np.all(abs(local[:2])<half[:2]-1e-5) and abs(local[2]-plane)<=.002+1e-6)
                            magnitude=float(np.linalg.norm([impulse.x,impulse.y,impulse.z])) if possible else 0.
                            self.max_internal_support_friction_impulse=max(self.max_internal_support_friction_impulse,magnitude)
                            if magnitude>self.max_internal_support_impulse:
                                self.max_internal_support_impulse=magnitude
                                self.internal_support_example=dict(layer=self.names[layer['geom']]+'/'+layer['layer'],
                                    point=[point.x,point.y,point.z],component='possibly_internal_friction',impulse_ns=magnitude)
                    self.friction.append((first,second,point.x,point.y,point.z,impulse.x,impulse.y,impulse.z))
                    self.friction_shapes.append(tuple(ids));self.extracted_friction+=1

    def update(self,q,v,command,dt, *, carried=False,grasp_contact_allowed=True):
        self.time=self.sim.time;near=self.proximity_data;near.qpos[:]=self.plant.configuration(q)
        # This independent model only supplies geometry. Inertia factorization,
        # tendon and constraint assembly from mj_fwdPosition are unused here.
        mujoco.mj_kinematics(self.proximity_model,near);mujoco.mj_collision(self.proximity_model,near)
        pos=near.xpos[self.object_body];R=near.xmat[self.object_body].reshape(3,3)
        oq=self.plant.meta['object']['q'];clearance=self.shape.lowest_z(q[oq:oq+7])-self.support_top
        self.max_clearance=max(self.max_clearance,clearance)
        if carried:self.min_clearance_carried=min(self.min_clearance_carried,clearance)
        self.max_joint_speed=max(self.max_joint_speed,float(abs(v[self.vi]).max()))
        raw=self.kp*(command-q[self.qi])-self.kd*v[self.vi]
        self.max_raw_effort_ratio=max(self.max_raw_effort_ratio,float((abs(raw)/self.effort).max()))
        self.saturated_time+=(abs(raw)>self.effort)*dt;self.max_servo_error=max(self.max_servo_error,float(abs(command-q[self.qi]).max()))
        self.max_joint_limit_violation=max(self.max_joint_limit_violation,float(np.maximum(self.limits[:,0]-q[self.qi],q[self.qi]-self.limits[:,1]).max()))
        n1,n2=np.asarray(near.contact.geom1),np.asarray(near.contact.geom2);ngap=np.asarray(near.contact.dist)
        self.max_nominal_geometry_penetration=max(self.max_nominal_geometry_penetration,float(-ngap.min(initial=0.)))
        no1,no2=n1==self.object_geom,n2==self.object_geom;other=np.where(no1,n2,n1)
        intended=(no1|no2)&(((self.groups[other]>=0)&grasp_contact_allowed)|self.allowed_support[other]);environment=self.allowed_support[n1]|self.allowed_support[n2]
        # The primitive/convex object surfaces coincide across both models.
        # Their table penetration must not wait for a contact-report event.
        # The ellipsoid uses a cooked union and cannot use its nominal analytic
        # surface as an exact independent penetration oracle.
        if self.shape.family!='ellipsoid':
            support_pair=(no1|no2)&self.allowed_support[other]
            self.max_independent_support_penetration=max(self.max_independent_support_penetration,float(-ngap[support_pair].min(initial=0.)))
            self.max_penetration=max(self.max_penetration,self.max_independent_support_penetration)
        limits=np.where(environment,ENVIRONMENT_CLEARANCE,PART_CLEARANCE)
        self.minimum_part_gap=min(self.minimum_part_gap,float(ngap[~intended&~environment].min(initial=PART_CLEARANCE)))
        self.minimum_environment_gap=min(self.minimum_environment_gap,float(ngap[~intended&environment].min(initial=ENVIRONMENT_CLEARANCE)))
        for i in np.flatnonzero(~intended&(ngap<limits-1e-6)):self.clearance_violations.add(tuple(sorted([self.names[n1[i]],self.names[n2[i]]])))
        forces=np.zeros((len(self.links),3));moments=forces.copy();centers=forces.copy();totals=np.zeros(len(self.links))
        rows=np.asarray(self.contacts,dtype=float).reshape(-1,12)
        self.last_contact_rows=rows
        self.last_contact_shapes=np.asarray(self.contact_shapes,dtype=np.uint64).reshape(-1,2)
        self.last_friction_rows=np.asarray(self.friction,dtype=float).reshape(-1,8)
        self.last_friction_shapes=np.asarray(self.friction_shapes,dtype=np.uint64).reshape(-1,2)
        if len(rows):
            first,second=rows[:,:2].astype(int).T;gap=rows[:,-1]
            self.max_penetration=max(self.max_penetration,float(-gap.min()))
            obj_first=first==self.object_geom;obj_second=second==self.object_geom;obj=obj_first|obj_second
            other=np.where(obj_first,second,first);groups=self.groups[other]
            active=np.linalg.norm(rows[:,5:8],axis=1)/dt>.05
            unexpected=~(obj&(((groups>=0)&grasp_contact_allowed)|self.allowed_support[other]))&((gap < -1e-5)|active)
            for index in np.flatnonzero(unexpected):self.other_contacts.add(tuple(sorted([self.names[first[index]],self.names[second[index]]])))
            chosen=obj&(groups>=0);groups=groups[chosen];selected=rows[chosen];points=selected[:,2:5];impulses=selected[:,5:8];normals=selected[:,8:11]
            # PhysX impulse/normal act on collider0. Convert to the gripper.
            force=impulses/dt*np.where(obj_first[chosen],-1.,1.)[:,None]
            bodies=np.array([self.link_bodies[link] for link in self.links],dtype=int);origins=near.xpos[bodies[groups]]
            np.add.at(forces,groups,force);np.add.at(moments,groups,np.cross(points-origins,force))
            weights=np.maximum(0.,np.einsum('ij,ij->i',impulses,normals)/dt)
            np.add.at(totals,groups,weights);np.add.at(centers,groups,weights[:,None]*((points-pos)@R))
        friction=self.last_friction_rows
        if len(friction):
            first,second=friction[:,:2].astype(int).T
            obj_first=first==self.object_geom;obj=obj_first|(second==self.object_geom)
            other=np.where(obj_first,second,first);groups=self.groups[other]
            active=np.linalg.norm(friction[:,5:8],axis=1)/dt>.05
            unexpected=~(obj&(((groups>=0)&grasp_contact_allowed)|self.allowed_support[other]))&active
            for index in np.flatnonzero(unexpected):self.other_contacts.add(tuple(sorted([self.names[first[index]],self.names[second[index]]])))
            chosen=obj&(groups>=0);groups=groups[chosen];selected=friction[chosen]
            force=selected[:,5:8]/dt*np.where(obj_first[chosen],-1.,1.)[:,None]
            bodies=np.array([self.link_bodies[link] for link in self.links],dtype=int)
            np.add.at(forces,groups,force)
            np.add.at(moments,groups,np.cross(selected[:,2:5]-near.xpos[bodies[groups]],force))
        self.last_forces=forces;self.last_moments=moments
        self.max_force=max(self.max_force,float(np.linalg.norm(forces,axis=1).max(initial=0)))
        self.max_moment=max(self.max_moment,float(np.linalg.norm(moments,axis=1).max(initial=0)))
        self.last_normal_loads=totals
        if np.any(totals>.05):self.last_contact_time=self.time
        if carried:
            self.gap_time=np.where(totals>.5,0.,self.gap_time+dt);self.longest_gap=max(self.longest_gap,float(self.gap_time.max(initial=0)))
            for i,name in enumerate(self.links):
                if totals[i]<=.5:continue
                center=centers[i]/totals[i]
                if name not in self.anchors:self.anchors[name]=center
                self.max_slip=max(self.max_slip,float(np.linalg.norm(center-self.anchors[name])))
                bid=self.link_bodies[name];rotation=near.xmat[bid].reshape(3,3);origin=near.xpos[bid]
                if name not in self.material_anchors:self.material_anchors[name]=((R@center+pos-origin)@rotation,center.copy())
                local,initial=self.material_anchors[name];current=(rotation@local+origin-pos)@R
                self.max_material_displacement=max(self.max_material_displacement,float(np.linalg.norm(current-initial)))
        self.contacts.clear();self.contact_shapes.clear();self.friction.clear();self.friction_shapes.clear()

    def result(self):
        result=super().result();result.pop('peak_effort_ratio')
        result.pop('support_representation',None)
        result['raw_servo_limit_exceedance_time_s']=result.pop('actuator_saturation_time_s')
        result.update(engine='PhysX',drive_effort_evidence='source maxForce checked; raw servo estimate logged, actual implicit drive effort not directly measured')
        result.update(contact_force_components='normal_and_friction',
                      clearance_detection='refitted_body_bounds',
                      reported_friction_anchors=self.reported_friction,extracted_friction_anchors=self.extracted_friction)
        result.update(penetration_metric='maximum of PhysX-reported penetration and independent non-ellipsoid object/support geometry; unchanged 2 mm acceptance threshold',
                      peak_independent_support_penetration_m=self.max_independent_support_penetration,
                      reported_contact_points=self.reported_points,extracted_contact_points=self.extracted_points,
                      support_layer_contact_points=self.support_layer_points,
                      peak_internal_support_boundary_impulse_ns=self.max_internal_support_impulse,
                      peak_internal_support_friction_impulse_ns=self.max_internal_support_friction_impulse,
                      support_boundary_force_components='normal_and_friction_conservative_anchor_geometry',
                      internal_support_boundary_example=self.internal_support_example,
                      peak_nominal_geometry_penetration_m=self.max_nominal_geometry_penetration,
                      nominal_geometry_scope='independent MuJoCo distance only; analytic ellipsoid, not the cooked union')
        return result
