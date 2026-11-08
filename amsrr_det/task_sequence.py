"""Continuous free-module assembly -> flight transfer -> existing task control.

One physical MuJoCo model/data persists throughout. Articulated controller
coordinates are read-only projections of observed joint coordinates; secondary
free roots and their residual seam motion remain in the physical simulation.
"""
import copy
import time
from pathlib import Path
import numpy as np
import mujoco
import pinocchio as pin
from lxml import etree as ET
from .task_execution import write
from .assembly import forest_posture,DockFilters
from .assembly_plan import prepare


class SequenceFailure(RuntimeError):
    task_failure=True


class Coordinates:
    def __init__(self,physical,control):
        self.physical,self.control=physical,control
        self.pq=[];self.cq=[];self.pv=[];self.cv=[]
        for j in range(control.njnt):
            name=control.joint(j).name
            try:k=physical.joint(name).id
            except KeyError:raise ValueError('missing physical joint '+name)
            if control.jnt_type[j]!=physical.jnt_type[k]:raise ValueError('joint type changed '+name)
            nq,nv=(7,6) if control.jnt_type[j]==0 else (1,1)
            self.pq.extend(physical.jnt_qposadr[k]+np.arange(nq));self.cq.extend(control.jnt_qposadr[j]+np.arange(nq))
            self.pv.extend(physical.jnt_dofadr[k]+np.arange(nv));self.cv.extend(control.jnt_dofadr[j]+np.arange(nv))
        self.pq,self.cq,self.pv,self.cv=map(lambda a:np.array(a,dtype=int),(self.pq,self.cq,self.pv,self.cv))
        if physical.nu!=control.nu:raise ValueError('actuator count mismatch')
        self.pu=np.array([physical.actuator(control.actuator(i).name).id for i in range(control.nu)])

    def observe(self,q,v):
        qc=self.control.qpos0.copy();vc=np.zeros(self.control.nv)
        qc[self.cq]=q[self.pq];vc[self.cv]=v[self.pv]
        return qc,vc


class BlendedControl:
    def __init__(self,controller,applied,mapping=None):
        self.mapping=mapping
        self.controller=controller;self.old=applied.copy() if mapping is None else applied[mapping.pu].copy();self.offset=None;self.start=None
        self.last_command=applied.copy();self.last={}
        self.relative_feedback=getattr(controller,"relative_feedback",False)
    def update(self,*args,**kwargs):
        command=self.controller.update(*args,**kwargs)
        now=kwargs.get('observation_time',0.)
        if self.start is None:self.start=now;self.offset=self.old-command
        u=float(np.clip((now-self.start)/.6,0.,1.));s=u**3*(10-15*u+6*u*u)
        value=command+(1-s)*self.offset
        if self.mapping is None:self.last_command=value
        else:self.last_command[self.mapping.pu]=value
        self.last=getattr(self.controller,'last',{})
        return self.last_command.copy()


def add_scene(plan,task_plant,task_data):
    """Append task environment before any physics; retain module geometry."""
    p=plan['plant'];doc=ET.fromstring(p.xml);source=ET.fromstring(task_plant.xml)
    asset=doc.find('asset');known={e.get('name') for e in asset}
    for e in source.find('asset'):
        if e.get('name') not in known:asset.append(copy.deepcopy(e));known.add(e.get('name'))
    world=doc.find('worldbody')
    for e in source.find('worldbody'):
        if e.tag=='light' or e.get('name')=='floor':continue
        # Robot world roots contain an mN/root body. Everything else is scene.
        if e.tag=='body' and any((b.get('name') or '').startswith('m') and (b.get('name') or '').endswith('/root') for b in e.iter('body')):continue
        node=copy.deepcopy(e)
        for b in node.iter('body'):
            if (b.get('name') or '').startswith('joint'):b.set('name','scene_'+b.get('name'))
        world.append(node)
    # Preserve task pad materials but keep the assembly position-drive encoding.
    geoms={g.get('name'):g for g in source.findall('.//geom')}
    for g in doc.findall('.//geom'):
        other=geoms.get(g.get('name'))
        if other is not None:
            for key in ['friction','solref','solimp','priority','condim']:
                if other.get(key) is not None:g.set(key,other.get(key))
    option=doc.find('option')
    option.set('integrator',['Euler','RK4','implicit','implicitfast','discrete'][int(task_plant.mj.opt.integrator)])
    for name in ['timestep','impratio','ccd_tolerance','ccd_iterations','iterations','tolerance']:
        option.set(name,str(getattr(task_plant.mj.opt,name)))
    old=p.mj;p.xml=ET.tostring(doc,encoding='unicode');p.mj=mujoco.MjModel.from_xml_string(p.xml)
    from .collision_policy import configure
    configure(p.mj)
    # New object coordinates are initialized from task inputs exactly once.
    initial=p.mj.qpos0.copy()
    for j in range(task_plant.mj.njnt):
        name=task_plant.mj.joint(j).name
        if name.startswith('m'):continue
        k=p.mj.joint(name).id;nq=7 if p.mj.jnt_type[k]==0 else 1
        initial[p.mj.jnt_qposadr[k]:p.mj.jnt_qposadr[k]+nq]=task_data.qpos[task_plant.mj.jnt_qposadr[j]:task_plant.mj.jnt_qposadr[j]+nq]
    def expand(q):
        v=initial.copy();v[:old.nq]=q;return v
    plan['initial']=expand(plan['initial']);plan['closed']=expand(plan['closed'])
    plan['poses']={k:expand(q) for k,q in plan['poses'].items()}
    plan['transit']=[(expand(a),expand(b),t) for a,b,t in plan['transit']]
    plan['filters']=DockFilters(p,plan['edges']);plan['filters'].apply(p.mj,[])
    return p


def switch_drives(physical,control,data,mapping):
    """Keep the commanded effort exactly continuous across drive encodings."""
    mujoco.mj_fwdPosition(physical,data);mujoco.mj_fwdVelocity(physical,data)
    oldraw=physical.actuator_gainprm[:,0]*data.ctrl+physical.actuator_biasprm[:,1]*data.actuator_length+physical.actuator_biasprm[:,2]*data.actuator_velocity
    for name in ['actuator_gainprm','actuator_biasprm']:
        getattr(physical,name)[mapping.pu]=getattr(control,name)
    data.ctrl[:]=(oldraw-physical.actuator_biasprm[:,1]*data.actuator_length-physical.actuator_biasprm[:,2]*data.actuator_velocity)/physical.actuator_gainprm[:,0]


class Sequence:
    def __init__(self,out,seed,observer,worker_cpu=2):
        self.out=Path(out);self.seed=seed;self.observer=observer;self.worker_cpu=worker_cpu;self.result=None;self.transfer_result=None;self.pauses=[];self.pending_wall=None;self.hold_publications=[]

    def start(self,task_plant,task_data):
        """Return the same live physical state after assembly and flight transfer."""
        from scripts.run_assembly_task import run
        case=dict(id='interactive',task='sequence',modules=task_plant.meta['modules'],edges=task_plant.meta['edges'],initial_seed=self.seed)
        before=time.perf_counter()
        try:plan=prepare(case,task_plant.mj.opt.timestep)
        except ValueError as error:raise SequenceFailure('assembly_planning: '+str(error)) from error
        physical=add_scene(plan,task_plant,task_data)
        write(self.out/'sequence_preparation.json',dict(seconds=time.perf_counter()-before,scope='additional assembly preparation'))
        answer=run(case,self.out/'assembly',worker_cpu=self.worker_cpu,observer=self.observer,prepared_plan=plan,return_context=True)
        result,context=answer if isinstance(answer,tuple) else (answer,None)
        self.result=result
        if not result.get('physical_success'):raise SequenceFailure('assembly_failed: '+result['reason'])
        d=context['data'];self.data=d;self.air=context['air'];self.physical=physical;self.map=Coordinates(physical.mj,task_plant.mj)
        self.pending_wall=context['boundary_wall']
        self.transfer(task_plant,task_data,context)
        self.resume(self.advance_plain,"transfer_cleanup")
        # Change only the supported drive encoding, preserving instantaneous
        # commanded effort. Bodies, velocities, time and seams are untouched.
        switch_drives(physical.mj,task_plant.mj,d,self.map)
        # Emit matching XML for future read-only display/audit reconstruction.
        doc=ET.fromstring(physical.xml);source=ET.fromstring(task_plant.xml)
        for node in doc.find('actuator'):
            other=source.find(f"actuator/*[@name='{node.get('name')}']")
            for key in ['gainprm','biasprm']:
                if other.get(key) is not None:node.set(key,other.get(key))
        physical.xml=ET.tostring(doc,encoding='unicode')
        np.savez(self.out/'task_handoff.npz',q=d.qpos,v=d.qvel,ctrl=d.ctrl,time=d.time,seams=d.eq_active)
        return physical,d

    def transfer(self,task_plant,target,context):
        """Drive articulated postures in the existing physical world."""
        from build.async_allocation import AsyncAllocation
        from build.control_latency import ControlCycle
        p=self.physical;m=p.mj;d=context['data'];control=context['control'];air=context['air']
        from .mujoco_model import make_plant
        robot=make_plant(task_plant.meta['modules'],task_plant.meta['edges'],meshes=False,compile_threads=False)
        task_to_robot=Coordinates(task_plant.mj,robot.mj)
        robot_target=mujoco.MjData(robot.mj);robot_target.qpos[:],robot_target.qvel[:]=task_to_robot.observe(target.qpos,target.qvel)
        task_plant=robot;target=robot_target
        mapping=Coordinates(m,robot.mj);qc,vc=mapping.observe(d.qpos,d.qvel)
        observed=mujoco.MjData(task_plant.mj);observed.qpos[:]=qc;observed.qvel[:]=vc
        start,_=task_plant.observation(observed);goal,_=task_plant.observation(target)
        high=max(start[2],goal[2])+0.6
        lift=start.copy();lift[2]=high
        morph=goal.copy();morph[:3]=lift[:3]
        over=goal.copy();over[2]=high
        points=[start,lift,morph,over,goal]
        durations=[]
        for a,b in zip(points[:-1],points[1:]):
            delta=pin.difference(task_plant.pin,a,b)
            durations.append(max(2.,1.875*np.linalg.norm(delta[:3])/.4,1.875*float(abs(delta[3:]).max(initial=0))/.6))
        control.last_command=d.ctrl.copy()
        allocation=AsyncAllocation(control,self.worker_cpu);calls=[]
        def physical_target(q):
            robot=forest_posture(p,task_plant,q)
            v=p.configuration(robot)
            # Preserve actual scene coordinates as references; never assign them.
            if m.nq>p.pin.nq:v[p.pin.nq:]=d.qpos[p.pin.nq:]
            return v
        advance=self.advance_plain
        self.resume(advance,'assembly_to_transfer')
        t0=float(d.time)
        failure=None
        try:
            end=sum(durations)
            while d.time-t0<end+3.:
                stamp=time.perf_counter();cycle=ControlCycle(m,d,.01);t=float(d.time)-t0
                i=min(np.searchsorted(np.cumsum(durations),t,side='right'),len(durations)-1)
                s=float(np.clip((t-sum(durations[:i]))/durations[i],0.,1.));f=s**3*(10-15*s+6*s*s)
                q=pin.interpolate(task_plant.pin,points[i],points[i+1],f);reference=physical_target(q)
                # The controller also uses position/attitude feedback; reference
                # derivatives are formed from neighboring target configurations.
                eps=.001
                def at(dt):
                    u=float(np.clip((t+dt-sum(durations[:i]))/durations[i],0.,1.));w=u**3*(10-15*u+6*u*u)
                    return physical_target(pin.interpolate(task_plant.pin,points[i],points[i+1],w))
                qa,qb=at(-eps),at(eps);velocity=np.zeros(m.nv);acc=np.zeros(m.nv)
                mujoco.mj_differentiatePos(m,velocity,2*eps,qa,qb)
                for k,j in zip(context['plan']['qi'],context['plan']['vi']):
                    R=pin.Quaternion(np.r_[reference[k+4:k+7],reference[k+3]]).matrix();velocity[j+3:j+6]=R@velocity[j+3:j+6]
                u=allocation.update(float(d.time),'TRANSFER_TO_TASK',(d.qpos.copy(),d.qvel.copy(),reference,velocity,acc,d.eq_active.copy()),dict(observation_time=float(d.time),applied_command=d.ctrl.copy()),observed_wall=stamp)
                if self.observer:self.observer.update(d,'TRANSFER_TO_TASK')
                elapsed=time.perf_counter()-stamp;failure=cycle.wait_for_result(elapsed,advance)
                if not failure:cycle.apply(u);failure=advance(cycle.remaining_steps())
                calls.append(elapsed)
                if failure:break
            self.pending_wall=time.perf_counter()
            final=mapping.observe(d.qpos,d.qvel)[0];delta=np.zeros(task_plant.mj.nv);mujoco.mj_differentiatePos(task_plant.mj,delta,1.,final,target.qpos)
            # Record tracking error without resetting the physical state.
            pose_error=float(np.linalg.norm(delta[:3]));joint_error=float(np.max(abs(delta[6:]),initial=0))
            self.transfer_result=dict(duration_s=float(d.time)-t0,position_error_m=pose_error,joint_error=joint_error,publication_max_s=max(calls,default=0),publication_over_10ms=sum(t>.01 for t in calls),failure=failure)
            write(self.out/'transfer.json',self.transfer_result)
            if failure:raise SequenceFailure(failure)
        finally:allocation.close()

    def advance_plain(self,n):
        m=self.physical.mj;d=self.data
        for _ in range(n):
            self.air.apply(d);mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all() or np.max(abs(d.qvel))>200:return 'extreme_instability'
            if d.ncon and np.min(d.contact.dist)<-.035:return 'extreme_collision'
        return None

    def advance_audited(self,audit,n,**kwargs):
        # The airflow model samples at 2 ms; native physics/audit stays batched.
        while n>0:
            self.air.apply(self.data)
            steps=min(n,max(1,int(np.ceil((self.air.next_update-self.data.time)/self.physical.mj.opt.timestep-1e-9))))
            failure=audit.advance(self.data,steps,**kwargs)
            if failure:return failure
            n-=steps
        return None

    def resume(self,advance,label='task_start'):
        if self.pending_wall is None:return
        elapsed=time.perf_counter()-self.pending_wall
        before=float(self.data.time)
        # Event-time fallback publisher repeats the already applied command
        # every 10 ms while preparation is pending. It uses no new observation.
        # Numerical integration is executed after measuring preparation, just
        # like ControlCycle; no future state is used to form these outputs.
        m=self.physical.mj;held=self.data.ctrl.copy();left=int(np.ceil(elapsed/m.opt.timestep));failure=None
        while left>0:
            stamp=time.perf_counter();self.data.ctrl[:]=held;cost=time.perf_counter()-stamp
            self.hold_publications.append(dict(stage=label,time_s=float(self.data.time),calculation_s=cost,new_observation=False))
            steps=min(left,max(1,round(.01/m.opt.timestep)))
            failure=advance(steps);left-=steps
            if failure:break
        self.pauses.append(dict(stage=label,measured_s=elapsed,from_s=before,to_s=float(self.data.time),failure=failure))
        write(self.out/'transition_delays.json',self.pauses)
        write(self.out/'transition_publications.json',self.hold_publications)
        self.pending_wall=time.perf_counter()
        if failure:raise SequenceFailure(failure)

    def reference(self,original,task_plant):
        sequence=self
        class PhysicalReference:
            def __getattr__(self,name):return getattr(original,name)
            def pose(self,t):
                q=original.pose(t)
                data=mujoco.MjData(task_plant.mj);data.qpos[:]=q
                pin_q,_=task_plant.observation(data)
                return sequence.physical.configuration(forest_posture(sequence.physical,task_plant,pin_q))
        return PhysicalReference()
