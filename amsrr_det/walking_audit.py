"""Independent physical evidence for walking, never a controller input."""
import numpy as np
import mujoco
from .walking import foot_material,requires_face_alignment


class WalkingAudit:
    def __init__(self,plant,links,spec,reference,*,start_time=0.):
        self.plant=plant;self.m=plant.mj;self.spec=spec;self.reference=reference
        self.start_time=float(start_time)
        self.face_alignment=np.array([requires_face_alignment(link) for link in links])
        self.links=links;self.material=foot_material(plant,links);self.count=len(links)
        m=self.m;self.floor=m.geom('floor').id;self.groups=np.full(m.nbody,-1,int)
        for i,link in enumerate(links):self.groups[m.body(link).id]=i
        self.effort=np.r_[[a['effort'] for a in plant.meta['actuators']],[r['maximum'] for r in plant.meta['rotors']]]
        self.nj=len(plant.meta['actuators']);self.weight=plant.meta['mass']*9.80665
        self.foot_impulse=np.zeros(self.count);self.other_impulse=0.;self.ground_time=0.;self.elapsed=0.
        self.energy_rotor=0.;self.energy_joint=0.;self.positive_work=0.
        self.rotor_impulse=0.;self.rotor_vertical_impulse=0.
        self.rotor_sites=np.array([m.site(plant.pin.frames[r['frame']].name).id for r in plant.meta['rotors']])
        self.peak_effort=0.;self.peak_penetration=0.;self.peak_speed=0.;self.abort=None
        self.samples=[];self.cycles={};self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy();self.delta=np.zeros(m.nv)
        self.force=np.zeros(6);self.slip_integral=0.;self.last_load=np.zeros(self.count)
        self.vi=np.array([m.jnt_dofadr[m.joint(a['name']).id] for a in plant.meta['actuators']])
        # Measured foot motion, independent of commanded gait phase. A foot
        # touching during flight or twitching in place is not a completed step.
        self.steps=[];self.foot_state=[dict(landing=None,airborne=False,clearance=0.,
            support_s=0.,impulse=0.,slip=0.,start_progress=0.,end_progress=0.,contact_s=0.) for _ in links]
        self.minimum_stride=np.array([max(.1,float(np.ptp(v,axis=0).max())) for _,v in self.material])
        self.last_observation_s=None;self.start_progress=None;self.progress=0.;self.last_angle=None;self.unwrapped_angle=0.
        self.face_angle_max=0.;self.face_bad_impulse=0.;self.face_front=np.array([v[:,0].max() for _,v in self.material]);self.dominant_support_s=0.;self.light_support_s=0.;self.longest_light_support_s=0.

    def reached(self):
        return self.start_progress is not None and self.progress-self.start_progress>=self.spec.distance

    def physics(self,d):
        m=self.m;dt=m.opt.timestep;self.elapsed+=dt
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():self.abort='nonfinite_physical_state';return
        ratio=float(np.max(abs(d.actuator_force)/self.effort));self.peak_effort=max(self.peak_effort,ratio)
        self.peak_speed=max(self.peak_speed,float(np.max(abs(d.qvel[self.vi]))))
        if ratio>1.00001:self.abort='motor_effort_limit'
        if np.linalg.norm(d.qvel[:3])>20 or np.linalg.norm(d.qvel[3:6])>50:self.abort='extreme_control_instability'
        if d.ncon:
            self.peak_penetration=max(self.peak_penetration,float(-np.min(d.contact.dist)))
            if np.min(d.contact.dist)<-.03:self.abort='extreme_collision'
        thrust=np.maximum(d.actuator_force[self.nj:],0.)
        self.energy_rotor+=float(thrust**1.5@np.ones(m.nu-self.nj))*dt
        self.rotor_impulse+=float(thrust.sum())*dt
        self.rotor_vertical_impulse+=float(thrust@d.site_xmat[self.rotor_sites,8])*dt
        self.energy_joint+=float(np.sum((d.actuator_force[:self.nj]/self.effort[:self.nj])**2))*dt
        self.positive_work+=float(np.maximum(d.actuator_force[:self.nj]*d.qvel[self.vi],0.).sum())*dt
        load=np.zeros(self.count);other=0.
        for j,contact in enumerate(d.contact):
            ga,gb=map(int,contact.geom)
            if self.floor not in (ga,gb):continue
            bid=int(m.geom_bodyid[gb if ga==self.floor else ga]);group=self.groups[bid]
            mujoco.mj_contactForce(m,d,j,self.force);normal=max(float(self.force[0]),0.)
            if group<0:other+=normal
            else:
                load[group]+=normal
                rotation=d.xmat[bid].reshape(3,3)
                cosine=-rotation[2,0]
                local=(contact.pos-d.xpos[bid])@rotation
                if self.face_alignment[group] and (cosine<.9961946980917455 or local[0]<self.face_front[group]-.003):
                    self.face_bad_impulse+=normal*dt
        # Apply the load threshold to a whole foot, not each solver contact:
        # subdividing one face into more contact points must not hide a tilt.
        for i in np.flatnonzero((load>.02*self.weight)&self.face_alignment):
            cosine=-d.xmat[self.material[i][0],6]
            if cosine<np.cos(self.face_angle_max):self.face_angle_max=float(np.arccos(np.clip(cosine,-1.,1.)))
        self.foot_impulse+=load*dt;self.other_impulse+=other*dt;self.last_load=load
        self.ground_time+=float(load.sum()>.005*self.weight)*dt
        if load.sum()>=.5*self.weight:
            self.dominant_support_s+=dt;self.light_support_s=0.
        else:
            self.light_support_s+=dt;self.longest_light_support_s=max(self.longest_light_support_s,self.light_support_s)
        phase=int(max(0.,d.time-self.start_time-dt)/self.reference.period)
        cycle=self.cycles.setdefault(phase,dict(time=0.,impulse=np.zeros(self.count),lift=np.full(self.count,-np.inf)))
        cycle['time']+=dt;cycle['impulse']+=load*dt

    def observe(self,d):
        """Kinematic/task evidence at control ticks; force peaks use every step."""
        m=self.m;now=float(d.time)-self.start_time;qr=self.reference.pose(now)
        mujoco.mj_differentiatePos(m,self.delta,1.,d.qpos,qr)
        # Separate data makes the observer unable to change the plant solver.
        if not hasattr(self,'observed'):self.observed=mujoco.MjData(m)
        od=self.observed;od.qpos[:]=d.qpos;od.qvel[:]=d.qvel
        mujoco.mj_kinematics(m,od);mujoco.mj_comPos(m,od)
        com=od.subtree_com[0].copy();goal=self.reference.origin[:2]+self.spec.path(now)[0]
        relative=com[:2]-self.reference.origin[:2];k=self.spec.curvature
        if abs(k)<1e-10:self.progress=float(relative[0])
        else:
            angle=float(np.arctan2(k*relative[0],1-k*relative[1]))
            if self.last_angle is None:self.unwrapped_angle=angle
            else:self.unwrapped_angle+=float(np.arctan2(np.sin(angle-self.last_angle),np.cos(angle-self.last_angle)))
            self.last_angle=angle;self.progress=self.unwrapped_angle/k
        if self.start_progress is None:self.start_progress=self.progress
        dt=0. if self.last_observation_s is None else now-self.last_observation_s
        self.last_observation_s=now
        lifts=[];slip=0.;foot_positions=[]
        for i,(bid,vertices) in enumerate(self.material):
            world=vertices@od.xmat[bid].reshape(3,3).T+od.xpos[bid];lifts.append(float(world[:,2].min()))
            mujoco.mj_jac(m,od,self.jp,self.jr,od.xpos[bid],bid)
            velocity=self.jp@d.qvel;omega=self.jr@d.qvel
            near=world[world[:,2]<=world[:,2].min()+.0005]
            speeds=velocity+np.cross(omega,near-od.xpos[bid])
            slip_speed=float(np.mean(np.linalg.norm(speeds[:,:2],axis=1)))
            slip+=self.last_load[i]*slip_speed
            point=od.xpos[bid].copy();foot_positions.append(point.tolist());state=self.foot_state[i]
            loaded=self.last_load[i]>.02*self.weight and lifts[-1]<.003
            state['clearance']=max(state['clearance'],lifts[-1])
            if lifts[-1]>=.02:state['airborne']=True
            if loaded:
                state['contact_s']+=dt
                if state['contact_s']>=.03 and state['airborne']:
                    previous=state['landing']
                    if previous is not None:
                        heading=self.spec.curvature*.5*(self.progress+state['start_progress'])
                        stride=float((point[:2]-previous[:2])@np.array([np.cos(heading),np.sin(heading)]))
                        advance=state['end_progress']-state['start_progress']
                        valid=(stride>=self.minimum_stride[i] and state['clearance']>=.02
                            and state['support_s']>=.05 and advance>=.1*stride
                            and state['slip']<=.2*stride)
                        self.steps.append(dict(foot=i,time_s=now,stride_m=stride,clearance_m=state['clearance'],
                            support_s=state['support_s'],body_advance_m=advance,stance_slip_m=state['slip'],valid=bool(valid)))
                    state.update(landing=point,airborne=False,clearance=0.,support_s=0.,impulse=0.,slip=0.,
                                 start_progress=self.progress,end_progress=self.progress)
                elif state['landing'] is None and state['contact_s']>=.03:
                    state['landing']=point;state['start_progress']=self.progress
                state['support_s']+=dt;state['impulse']+=self.last_load[i]*dt
                state['slip']+=slip_speed*dt;state['end_progress']=self.progress
            else:state['contact_s']=0.
        phase=int(max(0.,now-1e-9)/self.reference.period)
        if phase in self.cycles:self.cycles[phase]['lift']=np.maximum(self.cycles[phase]['lift'],lifts)
        record=dict(time_s=now,com=com.tolist(),path_error_m=float(np.linalg.norm(com[:2]-goal)),
            root_position_error_m=float(np.linalg.norm(self.delta[:3])),posture_error_rad=float(np.linalg.norm(self.delta[3:6])),
            foot_load_n=self.last_load.tolist(),foot_clearance_m=lifts,foot_positions_m=foot_positions,
            maximum_loaded_face_angle_deg=float(np.rad2deg(self.face_angle_max)),
            face_alignment_feet=[link for link,required in zip(self.links,self.face_alignment) if required],
            progress_m=self.progress-self.start_progress,weighted_slip_n_m_s=slip)
        self.samples.append(record);return record

    def result(self):
        if not self.samples:return dict(task_success=False,reason=self.abort or 'no_motion')
        duration=self.elapsed;period=self.reference.period
        complete=[r for r in self.cycles.values() if r['time']>=.99*period]
        good=np.array([(r['lift']>=.003)&(r['impulse']>=.005*self.weight*period) for r in complete])
        cycles=np.mean(good,axis=0) if len(good) else np.zeros(self.count)
        samples=self.samples;positions=np.array([r['com'][:2] for r in samples]);rel=positions-self.reference.origin[:2]
        if abs(self.spec.curvature)<1e-10:progress=rel[:,0]
        else:
            k=self.spec.curvature;progress=np.unwrap(np.arctan2(k*rel[:,0],1-k*rel[:,1]))/k
        times=np.array([r['time_s'] for r in samples]);average=(progress[-1]-progress[0])/max(times[-1]-times[0],1e-9)
        boundaries=np.r_[np.arange(0,times[-1],5.),times[-1]]
        distance=np.interp(boundaries,times,progress);window_speed=np.diff(distance)/np.diff(boundaries)
        support=float(self.foot_impulse.sum()/max(self.weight*duration,1e-9));ground=self.ground_time/max(duration,1e-9)
        other=self.other_impulse/max(self.weight*duration,1e-9);path=max(r['path_error_m'] for r in samples)
        posture=float(np.percentile([r['posture_error_rad'] for r in samples],95))
        minimum_steps=max(2,int(.8*(self.spec.distance/(self.spec.speed*period)-1)))
        step_counts=[sum(s['valid'] and s['foot']==i for s in self.steps) for i in range(self.count)]
        step_fractions=[sum(s['valid'] and s['foot']==i for s in self.steps)/max(1,sum(s['foot']==i for s in self.steps)) for i in range(self.count)]
        travel=self.progress-self.start_progress
        checks=dict(distance=travel>=self.spec.distance,physical=self.abort is None,
            minimum_speed=travel/max(duration,1e-9)>=.2,
            mean_speed=abs(average/self.spec.speed-1)<=.1,
            window_speed=bool(np.max(abs(window_speed/self.spec.speed-1))<=.2),
            path=path<=.1 and samples[-1]['path_error_m']<=.05,posture=posture<=.2,
            ground_support=support>=.5 and ground>=.9 and self.dominant_support_s/max(duration,1e-9)>=.5,
            face_alignment=self.face_angle_max<=np.deg2rad(5.) and self.face_bad_impulse<=.01*max(self.foot_impulse.sum(),1e-9),
            feet=bool(min(step_counts)>=minimum_steps and min(step_fractions)>=.8),other_support=other<=.02)
        checks={name:bool(value) for name,value in checks.items()}
        return dict(task_success=all(checks.values()),checks=checks,reason=self.abort,
            simulation_s=duration,distance_m=travel,mean_speed_m_s=average,window_speeds_m_s=window_speed.tolist(),
            maximum_path_error_m=path,posture_error_p95_rad=posture,ground_load_fraction=support,ground_contact_fraction=ground,
            foot_dominant_fraction=self.dominant_support_s/max(duration,1e-9),longest_light_support_s=self.longest_light_support_s,
            other_load_fraction=other,complete_cycles=len(complete),foot_cycle_success_fractions=cycles.tolist(),
            steps=self.steps,valid_steps_per_foot=step_counts,step_success_fractions=step_fractions,
            maximum_loaded_face_angle_deg=float(np.rad2deg(self.face_angle_max)),
            face_alignment_feet=[link for link,required in zip(self.links,self.face_alignment) if required],off_face_load_fraction=self.face_bad_impulse/max(self.foot_impulse.sum(),1e-9),
            minimum_stride_m=self.minimum_stride.tolist(),minimum_steps_per_foot=minimum_steps,
            rotor_burden_mean=self.energy_rotor/max(duration,1e-9),joint_burden_mean=self.energy_joint/max(duration,1e-9),
            rotor_thrust_weight_ratio=self.rotor_impulse/max(duration*self.weight,1e-9),
            rotor_vertical_weight_ratio=self.rotor_vertical_impulse/max(duration*self.weight,1e-9),
            rotor_burden_hover_ratio=self.energy_rotor/max(duration*self.weight**1.5/np.sqrt(len(self.rotor_sites)),1e-9),
            rotor_burden_per_m=self.energy_rotor/max(travel,1e-9),
            positive_joint_work_j=self.positive_work,peak_actuator_ratio=self.peak_effort,peak_penetration_m=self.peak_penetration,
            peak_joint_speed_rad_s=self.peak_speed)


class DriveAudit(WalkingAudit):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.dock=np.array([a['role']=='dock'for a in self.plant.meta['actuators']]);self.qi=self.vi+1;self.ranges=np.array([[a['lower'],a['upper']]for a in self.plant.meta['actuators']]);self.speed_limits=np.array([a.get('rated_speed') or a['speed']for a in self.plant.meta['actuators']]);self.speed_peak=np.zeros(self.nj);self.range_peak=np.zeros(self.nj);self.work_plus=np.zeros(self.nj);self.work_minus=np.zeros(self.nj);self.stop_peak=np.zeros(self.nj);self.stop_impulse=np.zeros(self.nj);self.stop_time=np.zeros(self.nj);self.joint_slots={self.m.joint(a["name"]).id:i for i,a in enumerate(self.plant.meta["actuators"])}
 def physics(self,d):
  super().physics(d);self.speed_peak=np.maximum(self.speed_peak,abs(d.qvel[self.vi]));q=d.qpos[self.qi];self.range_peak=np.maximum(self.range_peak,np.maximum(q-self.ranges[:,1],self.ranges[:,0]-q));power=d.actuator_force[:self.nj]*d.qvel[self.vi];self.work_plus+=np.maximum(power,0)*self.m.opt.timestep;self.work_minus+=np.maximum(-power,0)*self.m.opt.timestep
  indices=np.flatnonzero(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
  for row in indices:
   i=self.joint_slots.get(int(d.efc_id[row]))
   if i is not None:
    force=abs(float(d.efc_force[row]));self.stop_peak[i]=max(self.stop_peak[i],force);self.stop_impulse[i]+=force*self.m.opt.timestep;self.stop_time[i]+=self.m.opt.timestep
 def result(self):
  r=super().result();r['actuator_motion']={name:dict(max_speed_rad_s=float(self.speed_peak[mask].max()),max_speed_ratio=float((self.speed_peak/self.speed_limits)[mask].max()),max_range_excess_rad=float(self.range_peak[mask].max()),joint_limit_peak_nm=float(self.stop_peak[mask].max()),joint_limit_mean_absolute_nm=float(self.stop_impulse[mask].sum()/max(self.elapsed,1e-9)),max_joint_limit_time_fraction=float(self.stop_time[mask].max()/max(self.elapsed,1e-9)),positive_work_j=float(self.work_plus[mask].sum()),negative_work_j=float(self.work_minus[mask].sum()))for name,mask in [('dock',self.dock),('gimbal',~self.dock)]};return r
