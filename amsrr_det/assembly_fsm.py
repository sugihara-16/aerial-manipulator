"""Observed docking with retreat/retry and a 30-second assembly clock."""
import numpy as np
import mujoco
import pinocchio as pin
from .assembly import dock_error,ready_to_attach,following_base_target
from .assembly_plan import groups,eased


class AssemblyFSM:
    def __init__(self,plan):
        self.plan=plan;self.plant=plan['plant'];self.m=self.plant.mj
        self.qi=plan['qi'];self.vi=plan['vi'];self.n=len(self.qi);self.edges=plan['edges']
        self.stage='TAKEOFF';self.since=0.;self.clock=None;self.round=0;self.mask=0
        self.events=[];self.dwell=0.;self.retries=0;self.success=False;self.failure=None
        self.hover=plan['initial'].copy()
        for k in self.qi:self.hover[k+2]=1.5
        self.previous_time=0.;self.delta=np.zeros(self.m.nv)
        self.last_target=plan['initial'].copy();self.start=self.last_target.copy();self.end=self.hover.copy();self.duration=3.
        self.transit=[];self.motion_index=0;self.attach_dwell={};self.errors={}
        self.pdata=self.plant.pin.createData();self.corrections={};self.dt=.01
        self.initial_hover_since=None;self.control_ready=False;self.hover_samples=[];self.hover_observations=[];self.stable_since=None;self.capture_since={};self.capture_samples={};self.brake_coefficients=None

    def switch(self,stage,now,**info):
        self.events.append(dict(t=float(now),assembly_s=None if self.clock is None else float(now-self.clock),stage=stage,**info))
        self.stage=stage;self.since=float(now);self.dwell=0.

    def motion(self,a,b,now,duration):
        self.start=a.copy();self.end=b.copy();self.duration=duration;self.since=now;self.brake_coefficients=None

    def interpolate(self,now):
        s,ds,dds=eased(0.,1.,now-self.since,self.duration)
        mujoco.mj_differentiatePos(self.m,self.delta,1.,self.start,self.end)
        q=self.start.copy()
        if self.brake_coefficients is None:
            mujoco.mj_integratePos(self.m,q,self.delta,float(s));v=self.delta*ds;a=self.delta*dds
        else:
            s=float(np.clip((now-self.since)/self.duration,0.,1.));C=self.brake_coefficients
            x=sum(C[i]*s**(i+1) for i in range(5))
            v=sum((i+1)*C[i]*s**i for i in range(5))/self.duration
            a=sum((i+1)*i*C[i]*s**(i-1) for i in range(1,5))/self.duration**2
            mujoco.mj_integratePos(self.m,q,x,1.)
        # Return world-frame angular reference derivatives for each base.
        for k,j in zip(self.qi,self.vi):
            R=pin.Quaternion(np.r_[q[k+4:k+7],q[k+3]]).matrix()
            v[j+3:j+6]=R@v[j+3:j+6];a[j+3:j+6]=R@a[j+3:j+6]
        return q,v,a

    def begin_braking(self,data,target,velocity,acceleration,now):
        # Follow a continuous reference toward the observed stopping position.
        # All members of an existing component receive the same translation.
        end=target.copy();components=groups(self.n,self.edges,self.mask)
        for group in sorted(set(components)):
            members=[i for i,g in enumerate(components)if g==group]
            lead=min(members);k=self.qi[lead];j=self.vi[lead]
            shift=data.qpos[k:k+2]+.25*data.qvel[j:j+2]-target[k:k+2]
            for member in members:end[self.qi[member]:self.qi[member]+2]+=shift
        # Braking must also open the closing seam; holding its current
        # axial gap can let sideways drift hit a rotor arm before stopping.
        for edge in self.plan['rounds'][self.round]['edges']:
            if self.mask & (1 << edge):continue
            follower=self.edges[edge]['b'];extra=max(0.,.25-self.errors[edge]['axial_m'])
            for member in range(self.n):
                if member in self.plan['descendants'][edge]:
                    k=self.qi[member];end[k:k+3]+=extra*self.plan['directions'][edge]
        travel=max(np.linalg.norm(end[k:k+3]-target[k:k+3])for k in self.qi)
        self.motion(target,end,now,max(1.,1.875*travel/.25,np.sqrt(5.78*travel/.6)))
        delta=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,delta,1.,target,end)
        v=velocity.copy();a=acceleration.copy()
        for k,j in zip(self.qi,self.vi):
            R=pin.Quaternion(np.r_[target[k+4:k+7],target[k+3]]).matrix()
            v[j+3:j+6]=R.T@v[j+3:j+6];a[j+3:j+6]=R.T@a[j+3:j+6]
        v*=self.duration;a*=self.duration**2
        self.brake_coefficients=np.array([v,.5*a,10*delta-6*v-1.5*a,-15*delta+8*v+1.5*a,6*delta-3*v-.5*a])
        self.corrections={};self.switch('BRAKE',now)

    def speed(self,data):
        return max(np.linalg.norm(data.qvel[j:j+3]) for j in self.vi)

    def begin_round(self,now):
        if self.round==len(self.plan['rounds']):
            self.switch('VERIFY',now);return
        r=self.plan['rounds'][self.round]
        self.plan['filters'].apply(self.m,[i for i in range(len(self.edges))if self.mask&(1<<i)]+r['edges'])
        self.motion(self.last_target,self.plan['poses'][r['start_mask']],now,.7)
        self.attach_dwell={i:0. for i in r['edges']}
        self.capture_since={};self.capture_samples={i:[] for i in r['edges']}
        self.switch('ALIGN',now)

    def build_transit(self):
        self.transit=self.plan['transit']

    def relative_target(self,target,velocity,data,active):
        q,v=self.plant.observation(data);components=groups(self.n,self.edges,self.mask)
        for i in active:
            e=self.edges[i];b=e['b'];k=self.qi[b]
            # Slow outer-loop registration of the measured docking frames.
            # Directly feeding hinge vibrations (and their differentiated
            # velocity) back into the base reference excites the flexible
            # dock/body dynamics. Track their low-frequency displacement.
            a=e['a'];qa=self.qi[a]
            baseline=self.plan['closed']
            nominal_delta=(target[k:k+3]-target[qa:qa+3])-(baseline[k:k+3]-baseline[qa:qa+3])
            gap=float(nominal_delta@self.plan['directions'][i])
            desired=following_base_target(self.plant,q,e,max(0.,gap),self.pdata)
            R=pin.Quaternion(desired[3:7]).matrix();Rb=pin.Quaternion(np.r_[target[k+4:k+7],target[k+3]]).matrix()
            request=np.r_[desired[:3]-target[k:k+3],pin.log3(R@Rb.T)]
            previous=self.corrections.get(i,np.zeros(6))
            change_amount=(1-np.exp(-self.dt/.3))*(request-previous)
            limits=np.r_[np.full(3,.1),np.full(3,.3)]*self.dt
            updated=previous+np.clip(change_amount,-limits,limits)
            rate=(updated-previous)/max(self.dt,1e-6);self.corrections[i]=updated
            change=pin.exp3(updated[3:]);origin=target[k:k+3].copy();new_origin=origin+updated[:3]
            vb=velocity[self.vi[b]:self.vi[b]+3].copy()
            for node in range(self.n):
                if components[node]!=components[b]:continue
                z=self.qi[node];target[z:z+3]=new_origin+change@(target[z:z+3]-origin)
                old=pin.Quaternion(np.r_[target[z+4:z+7],target[z+3]]).matrix();qq=pin.Quaternion(change@old).coeffs()
                target[z+3:z+7]=np.r_[qq[3],qq[:3]]
                j=self.vi[node]
                velocity[j:j+3]=vb+rate[:3]+change@(velocity[j:j+3]-vb)+np.cross(rate[3:],target[z:z+3]-new_origin)
                velocity[j+3:j+6]=rate[3:]+change@velocity[j+3:j+6]
        return target

    def update(self,data):
        now=float(data.time);dt=now-self.previous_time;self.previous_time=now;self.dt=max(dt,1e-6)
        if self.clock is not None and now-self.clock>=30.-1e-8:
            self.failure='assembly_timeout'
            return self.last_target.copy(),np.zeros(self.m.nv),np.zeros(self.m.nv)
        target,vel,acc=self.interpolate(now)
        if self.stage=='TAKEOFF':
            errors=[np.linalg.norm(data.qpos[k:k+3]-self.hover[k:k+3])for k in self.qi]
            angular=max(np.linalg.norm(data.qvel[j+3:j+6])for j in self.vi)
            attitude=max(2*np.arccos(np.clip(abs(data.qpos[k+3:k+7]@self.hover[k+3:k+7]),0.,1.))for k in self.qi)
            good=max(errors)<.04 and self.speed(data)<.06 and angular<.1 and attitude<np.deg2rad(3.)
            if good:
                if self.initial_hover_since is None:self.initial_hover_since=now
            else:self.initial_hover_since=None
            self.dwell=0. if self.initial_hover_since is None else now-self.initial_hover_since
            if self.dwell>=.3-1e-9:
                self.clock=now;self.build_transit();self.switch('TRANSIT',now,qpos=data.qpos.tolist(),qvel=data.qvel.tolist())
                a,b,t=self.transit[0];self.motion(a,b,now,t)
            elif now>15.:self.failure='hover_not_acquired'
        elif self.stage=='TRANSIT' and now-self.since>=self.duration:
            self.motion_index+=1
            if self.motion_index<len(self.transit):
                a,b,t=self.transit[self.motion_index];self.motion(a,b,now,t)
            else:self.begin_round(now)
        elif self.stage in ('ALIGN','APPROACH','RETREAT'):
            r=self.plan['rounds'][self.round];q,v=self.plant.observation(data)
            active=[i for i in r['edges'] if not self.mask&(1<<i)]
            self.errors={i:dock_error(self.plant,q,v,self.edges[i],self.pdata) for i in active}
            if self.stage!='RETREAT':target=self.relative_target(target,vel,data,active)
            if self.stage=='ALIGN':
                good=(now-self.since>=self.duration and self.speed(data)<.08 and
                      all(e['transverse_m']<.012 and e['attitude_rad']<np.deg2rad(2.) and e['linear_speed_m_s']<.08 for e in self.errors.values()))
                self.dwell=self.dwell+dt if good else 0.
                if self.dwell>.15:
                    self.motion(self.plan['poses'][r['start_mask']],self.plan['poses'][r['end_mask']],now,r['duration'])
                    self.switch('APPROACH',now)
            elif self.stage=='APPROACH':
                unsafe=any(e['transverse_m']+.3*e['transverse_speed_m_s']>.015 or e['attitude_rad']>np.deg2rad(4.) for e in self.errors.values())
                if unsafe and now-self.since>.15:
                    self.retries+=1
                    if self.retries>3:self.failure='recovery_exhausted'
                    else:
                        self.begin_braking(data,target,vel,acc,now)
                else:
                    for i,error in self.errors.items():
                        if ready_to_attach(error):
                            self.capture_since.setdefault(i,now)
                            self.capture_samples[i].append(dict(t=now,**error))
                        else:
                            self.capture_since.pop(i,None);self.capture_samples[i]=[]
                        if i in self.capture_since and now-self.capture_since[i]>=.1-1e-9:
                            data.eq_active[i]=1;self.mask|=1<<i
                            self.events.append(dict(t=now,event='latch',edge=i,errors=error.copy(),qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),samples=self.capture_samples[i]))
                    if all(self.mask&(1<<i) for i in r['edges']):
                        self.motion(target,target,now,.6);self.switch('TRANSFER',now)
            elif now-self.since>=self.duration and self.speed(data)<.08 and max(np.linalg.norm(data.qvel[j+3:j+6])for j in self.vi)<.1:self.begin_round(now)
        elif self.stage=='BRAKE':
            good=now-self.since>=self.duration and self.speed(data)<.06 and max(np.linalg.norm(data.qvel[j+3:j+6])for j in self.vi)<.1
            self.dwell=self.dwell+dt if good else 0.
            if self.dwell>=.2:
                self.motion(target,self.plan['poses'][self.mask],now,1.)
                self.switch('RETREAT',now)
        elif self.stage=='TRANSFER' and now-self.since>=.6 and self.control_ready:
            self.round+=1;self.begin_round(now)
        elif self.stage=='VERIFY':
            q,v=self.plant.observation(data)
            self.errors={i:dock_error(self.plant,q,v,e,self.pdata)for i,e in enumerate(self.edges)}
            position=max(np.linalg.norm(data.qpos[k:k+3]-target[k:k+3])for k in self.qi)
            attitude=max(2*np.arccos(np.clip(abs(data.qpos[k+3:k+7]@target[k+3:k+7]),0.,1.))for k in self.qi)
            held=(self.control_ready and all(data.eq_active) and self.speed(data)<.06 and position<.04 and attitude<np.deg2rad(3.) and
                  all(abs(e['axial_m'])<.003 and e['transverse_m']<.002 and
                      e['attitude_rad']<np.deg2rad(.5) for e in self.errors.values()) and
                  max(np.linalg.norm(data.qvel[j+3:j+6])for j in self.vi)<.1)
            if held:
                if self.stable_since is None:self.stable_since=now
            else:self.stable_since=None
            self.dwell=0. if self.stable_since is None else now-self.stable_since
            self.hover_samples.append(dict(t=now,held=bool(held),position_m=float(position),attitude_rad=float(attitude),speed_m_s=float(self.speed(data)),central_ready=bool(self.control_ready)))
            self.hover_observations.append((now,data.qpos.copy(),data.qvel.copy(),target.copy(),bool(self.control_ready)))
            self.hover_observations=self.hover_observations[-110:]
            if self.dwell>=1.-1e-9:
                self.success=True
                self.events.append(dict(t=now,event='stable_central_hover',dwell_s=self.dwell,samples=self.hover_samples[-110:]))
        self.last_target=target.copy()
        return target,vel,acc
