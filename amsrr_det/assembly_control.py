"""Per-module flight allocation with bounded physical thrust and gimbals.

Before docking these are the issued single-module commands. After docking
the local feedback supplies desired body accelerations to one articulated
motion projection and rotor allocation. Real seams and joint reactions remain
in the plant.
"""
import time
import numpy as np
import pinocchio as pin
import mujoco
from .bounded_qp import bounded_qp
from .flight import cross3


class AssemblyControl:
    def __init__(self, plant, reference):
        self.plant=plant;self.model=plant.pin;self.meta=plant.meta
        self.data=self.model.createData();self.observed=mujoco.MjData(plant.mj)
        self.target_data=mujoco.MjData(plant.mj)
        act=self.meta['actuators'];rotors=self.meta['rotors'];self.nj=len(act);self.nr=len(rotors)
        self.qi=np.array([a['q'] for a in act]);self.vi=np.array([a['v'] for a in act])
        self.gi=np.array([r['gimbal'] for r in rotors])
        self.kp=np.array([a['kp'] for a in act]);self.kd=np.array([a['kd'] for a in act])
        self.limits=np.array([[a['lower'],a['upper']] for a in act]);self.speed=np.array([a['speed'] for a in act])
        self.maximum=np.array([r['maximum'] for r in rotors]);self.dock=np.array([a['role']=='dock' for a in act])
        self.frames=[r['frame'] for r in rotors];self.gj=[act[i]['joint'] for i in self.gi]
        self.observed.qpos[:]=reference;q,_=plant.observation(self.observed)
        self.command=q[self.qi].copy();self.force=np.full(self.nr,self.meta['mass']*9.80665/self.nr)
        self.last_command=np.r_[self.command,self.force];self.bias=np.zeros((len(self.meta['bases']),6))
        self.last_time=None;self.last={};self.trim=None;self.bandwidth=1.
        self.groups=[]
        for b in self.meta['bases']:
            ids=np.flatnonzero(np.array([r['module'] for r in rotors])==b['module'])
            n=len(ids);self.groups.append((b,ids,np.r_[ids,self.nr+ids],np.r_[np.full(n,5.),np.full(n,.5)],np.c_[np.eye(2*n),-np.eye(2*n)]))

    def update(self,qpos,qvel,reference,reference_velocity,reference_acceleration,seams,*,observation_time=None,deadline=None,recovery=False,allocate_modules=None):
        if deadline is None:deadline=time.perf_counter()+.008
        self.observed.qpos[:]=qpos;self.observed.qvel[:]=qvel
        q,v=self.plant.observation(self.observed)
        self.target_data.qpos[:]=reference;self.target_data.qvel[:]=0.
        target,_=self.plant.observation(self.target_data)
        m,d=self.model,self.data
        pin.forwardKinematics(m,d,q,v);pin.updateFramePlacements(m,d)
        dt=.01 if self.last_time is None or observation_time is None else np.clip(observation_time-self.last_time,0.,.05)
        self.last_time=observation_time
        self.bandwidth+=(1-np.exp(-dt/.2))*((.4 if recovery else 1.)-self.bandwidth)
        K=16*self.bandwidth**2;D=8*self.bandwidth
        acceleration=np.zeros(m.nv)
        for k,b in enumerate(self.meta['bases']):
            j,i=b['v'],b['q'];R=d.oMi[b['joint']].rotation;Rd=pin.Quaternion(target[i+3:i+7]).matrix()
            error=np.r_[target[i:i+3]-q[i:i+3],pin.log3(Rd@R.T)]
            self.bias[k]=np.clip(self.bias[k]+dt*np.where(abs(error)<.05,error,0.),-.3,.3)
            # Reference linear/angular velocities are world axes, indexed by
            # physical free joints. Pinocchio uses body coordinates.
            jid=self.plant.mj.joint(m.names[b['joint']]).id;vj=self.plant.mj.jnt_dofadr[jid]
            tv=reference_velocity[vj:vj+6];ta=reference_acceleration[vj:vj+6]
            acceleration[j:j+3]=R.T@(ta[:3]+K*error[:3]+D*(tv[:3]-R@v[j:j+3])+.7*self.bias[k,:3])-cross3(v[j+3:j+6],v[j:j+3])
            acceleration[j+3:j+6]=R.T@(ta[3:]+K*error[3:]+D*(tv[3:]-R@v[j+3:j+6])+.7*self.bias[k,3:])
        required=pin.rnea(m,d,q,v,acceleration).copy()
        positions=np.array([d.oMf[f].translation for f in self.frames]);directions=np.array([d.oMf[f].rotation[:,2] for f in self.frames])
        axes=np.array([d.oMi[j].rotation[:,0] for j in self.gj]);pivots=np.array([d.oMi[j].translation for j in self.gj])
        # Allocate over the servo's response time, while issuing at 100 Hz.
        # A 10 ms direction prediction asks the slow position servos for an
        # excessive change to correct an instantaneous base-wrench error.
        inputs=np.r_[self.force,self.command[self.gi]];ag=1-(1-np.exp(-1.))
        fallback=False;iterations=0;group_status=[]
        self.acceleration_requests={}
        for b,ids,cols,scale,C in self.groups:
            R=d.oMi[b['joint']].rotation;j=b['v'];origin=q[b['q']:b['q']+3]
            wrench_target=np.r_[R@required[j:j+3],R@required[j+3:j+6]]
            self.acceleration_requests[b['module']]=np.r_[R@acceleration[j:j+3]+cross3(R@v[j+3:j+6],R@v[j:j+3]),R@acceleration[j+3:j+6]]
            if allocate_modules is not None and b['module'] not in allocate_modules:continue
            lo=np.r_[np.zeros(len(ids)),np.maximum(self.limits[self.gi[ids],0]+.02,self.command[self.gi[ids]]-self.speed[self.gi[ids]]*.01)]
            hi=np.r_[self.maximum[ids],np.minimum(self.limits[self.gi[ids],1]-.02,self.command[self.gi[ids]]+self.speed[self.gi[ids]]*.01)]
            active_gi=self.gi[ids]
            qg=q[self.qi[active_gi]];vg=v[self.vi[active_gi]]
            effort=np.array([self.meta['actuators'][int(i)]['effort'] for i in active_gi])
            previous_effort=self.kp[active_gi]*(self.command[active_gi]-qg)-self.kd[active_gi]*vg
            guarded_low=qg+(np.minimum(-effort,previous_effort)+self.kd[active_gi]*vg)/self.kp[active_gi]
            guarded_high=qg+(np.maximum(effort,previous_effort)+self.kd[active_gi]*vg)/self.kp[active_gi]
            lo[len(ids):]=np.maximum(lo[len(ids):],guarded_low)
            hi[len(ids):]=np.minimum(hi[len(ids):],guarded_high)
            if np.any(lo>hi):
                group_status.append('bounds');fallback=True;continue
            failed=False
            for _ in range(2):
                angle=ag*(inputs[self.nr+ids]-q[self.qi[self.gi[ids]]]);c=np.cos(angle)[:,None];s=np.sin(angle)[:,None];axis=axes[ids]
                def rotate(x):return c*x+s*cross3(axis,x)+(1-c)*axis*np.sum(axis*x,axis=1)[:,None]
                direction=rotate(directions[ids]);arm=pivots[ids]+rotate(positions[ids]-pivots[ids])-origin
                force=inputs[ids,None]*direction;wrench=np.r_[force.sum(axis=0),cross3(arm,force).sum(axis=0)]
                df=ag*inputs[ids,None]*cross3(axis,direction);dp=ag*cross3(axis,rotate(positions[ids]-pivots[ids]))
                B=np.vstack([np.hstack([direction.T,df.T]),np.hstack([cross3(arm,direction).T,(cross3(dp,force)+cross3(arm,df)).T])])
                A=B*scale;rhs=wrench_target-wrench+B@inputs[cols]
                preferred=np.r_[np.full(len(ids),self.meta['mass']*9.80665/self.nr),np.zeros(len(ids))]/scale
                low,high=lo/scale,hi/scale
                # Each independent small allocation gets a bounded work slot.
                # A late observation must not starve the last module forever.
                # This worker never blocks the separate 100 Hz publisher.
                x,fall,it=bounded_qp(A.T@A+np.eye(len(cols))*.01,A.T@rhs+.01*preferred,C,np.r_[low,-high],low,high,inputs[cols]/scale,deadline=time.perf_counter()+.0005)
                inputs[cols]=x*scale;fallback|=fall;failed|=fall;iterations+=it
            group_status.append('expired' if failed else 'solved')
        self.force=inputs[:self.nr];self.command[self.gi]=inputs[self.nr:]
        if self.trim is None:
            ext=[pin.Force.Zero() for _ in range(m.njoints)]
            for rotor,f in zip(self.meta['rotors'],self.force):
                frame=m.frames[rotor['frame']];ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
            self.trim=pin.rnea(m,d,q,np.zeros(m.nv),np.zeros(m.nv),ext)[self.vi].copy()
        self.command[self.dock]=np.clip(target[self.qi[self.dock]]+self.trim[self.dock]/self.kp[self.dock],self.limits[self.dock,0],self.limits[self.dock,1])
        self.last_command=np.r_[self.command,self.force]
        self.last=dict(fallback=bool(fallback),iterations=iterations,seams=int(np.sum(seams)),groups=group_status)
        return self.last_command.copy()
