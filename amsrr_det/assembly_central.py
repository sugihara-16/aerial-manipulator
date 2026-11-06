"""A leader-owned whole-component flight controller and bumpless handover.

The physical free bodies and dock seams never change. Connected Pinocchio
views only provide the articulated model for the shared actuator allocation.
The lowest numbered member owns each component's allocation in this PC model;
the other members receive their joint and rotor commands from that allocation.
"""
import numpy as np
import mujoco
from .mujoco_model import make_plant
from .airborne_control import AirborneControl
from .assembly_control import AssemblyControl
from .assembly_plan import groups
import pinocchio as pin


def seam_mask(seams):
    return sum(1 << i for i,active in enumerate(seams) if active)


class CentralAssemblyControl:
    def __init__(self,plan):
        self.plan=plan;self.plant=plan['plant'];self.edges=plan['edges']
        self.single=AssemblyControl(self.plant,plan['initial'])
        self.nj=self.single.nj
        self.last_command=self.single.last_command.copy();self.last={};self.mask=0
        # Include every possible capture order within each parallel round.
        masks={0}
        for r in plan['rounds']:
            for subset in range(1 << len(r['edges'])):
                masks.add(r['start_mask'] | sum(1 << e for j,e in enumerate(r['edges']) if subset & (1 << j)))
        self.views={}
        for mask in sorted(masks-{0}):
            view=make_plant(self.plant.meta['modules'],[e for i,e in enumerate(self.edges) if mask & (1 << i)],meshes=False,compile_threads=False)
            ctrl=AirborneControl(view)
            q_source=[];q_dest=[];v_source=[];v_dest=[]
            for jid in range(view.mj.njnt):
                name=view.mj.joint(jid).name;src=self.plant.mj.joint(name).id
                nq,nv=(7,6) if view.mj.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE else (1,1)
                q_source.extend(self.plant.mj.jnt_qposadr[src]+np.arange(nq));q_dest.extend(view.mj.jnt_qposadr[jid]+np.arange(nq))
                v_source.extend(self.plant.mj.jnt_dofadr[src]+np.arange(nv));v_dest.extend(view.mj.jnt_dofadr[jid]+np.arange(nv))
            mapping=tuple(np.array(x) for x in (q_source,q_dest,v_source,v_dest))
            indices=np.array([self.plant.mj.actuator(view.mj.actuator(i).name).id for i in range(view.mj.nu)]);members=groups(view.meta['modules'],self.edges,mask)
            leaders={i:min(j for j,g in enumerate(members) if g==members[i]) for i in range(len(members))}
            controlled=np.array([sum(g==members[i] for g in members)>1 for i in range(len(members))])
            joint_modules=[int(a['name'].split('/')[0][1:]) for a in view.meta['actuators']]
            active=np.r_[controlled[joint_modules],controlled[[r['module'] for r in view.meta['rotors']]]]
            self.views[mask]=(view,ctrl,indices,active,leaders,mapping)

    def update(self,qpos,qvel,reference,velocity,acceleration,seams,*,applied_command=None,observation_time=None,deadline=None,recovery=False):
        mask=seam_mask(seams)
        components=groups(self.plant.meta['modules'],self.edges,mask)
        isolated={i for i,g in enumerate(components) if components.count(g)==1}
        # Isolated vehicles use the small flight allocation. Connected members
        # send body motion requests to ONE shared articulated controller.
        output=self.single.update(qpos,qvel,reference,velocity,acceleration,seams,
                                  observation_time=observation_time,deadline=deadline,recovery=recovery,allocate_modules=isolated)
        if not mask:
            self.last_command=output;self.last=dict(self.single.last,central_mask=0,leaders={i:i for i in range(self.plant.meta['modules'])})
            return output.copy()
        view,ctrl,indices,active,leaders,mapping=self.views[mask]
        qs,qd,vs,vd=mapping
        q=view.mj.qpos0.copy();q[qd]=qpos[qs];v=np.zeros(view.mj.nv);v[vd]=qvel[vs]
        target=view.mj.qpos0.copy();target[qd]=reference[qs]
        tv=np.zeros(view.mj.nv);ta=tv.copy();tv[vd]=velocity[vs];ta[vd]=acceleration[vs]
        for j in ctrl.base:
            jid=next(i for i in range(view.mj.njnt) if view.mj.jnt_dofadr[i]==j)
            k=view.mj.jnt_qposadr[jid];R=pin.Quaternion(np.r_[q[k+4:k+7],q[k+3]]).matrix()
            tv[j+3:j+6]=R.T@tv[j+3:j+6];ta[j+3:j+6]=R.T@ta[j+3:j+6]
        if mask!=self.mask:
            previous=self.last_command if applied_command is None else applied_command
            ctrl.seed(q,v,previous[indices]);self.mask=mask
        if applied_command is not None:ctrl.track_applied(applied_command[indices])
        central=ctrl.update(q,v,target,tv,ta,observation_time,nominal_input=output[indices],feedback_bandwidth=self.single.bandwidth,acceleration_requests=self.single.acceleration_requests)
        output[indices[active]]=central[active]
        self.last_command=output.copy()
        self.last=dict(central_mask=mask,leaders=leaders,central_groups=len(set(leaders.values())),
                       allocation=ctrl.last)
        return output


class SmoothHandoff:
    """Fade a handover bias while retaining the live new feedback."""
    def __init__(self,plant,edges,duration=.6):
        self.edges=edges;self.n=plant.meta['modules'];self.duration=duration;self.mask=0
        self.pending=[];self.events=[]
        acts=plant.meta['actuators'];rotors=plant.meta['rotors']
        self.modules=np.r_[[int(a['name'].split('/')[0][1:]) for a in acts],[r['module'] for r in rotors]]
        self.lower=np.r_[[a["lower"] for a in acts],np.zeros(len(rotors))]
        self.upper=np.r_[[a["upper"] for a in acts],[r["maximum"] for r in rotors]]

    def note(self,seams,applied,now):
        mask=seam_mask(seams)
        if mask==self.mask:return
        components=groups(self.n,self.edges,mask)
        joined={components[e[k]] for i,e in enumerate(self.edges) if mask & ~self.mask & (1 << i) for k in ('a','b')}
        members=[i for i,g in enumerate(components) if g in joined]
        active=np.isin(self.modules,members)
        self.pending=[p for p in self.pending if not np.any(p['active'] & active)]
        self.pending.append(dict(start=None,old=applied.copy(),active=active,first=True,event=len(self.events)))
        self.events.append(dict(t=now,mask=mask,members=members,leaders=sorted({min(j for j,g in enumerate(components) if g==components[i]) for i in members}),command_before=applied.tolist()))
        self.mask=mask

    def command(self,requested,applied,now):
        result=requested.copy();active=np.zeros(len(result),dtype=bool)
        for p in self.pending:
            if p['first']:
                p['start']=now;p['offset']=applied-requested
            s=float(np.clip((now-p['start'])/self.duration,0.,1.));w=s*s*s*(10+s*(-15+6*s))
            ids=p['active'];result[ids]=requested[ids]+(1-w)*p['offset'][ids];active|=ids
            if p['first']:
                self.events[p['event']]['first_output_difference']=float(np.max(abs(result[ids]-applied[ids])))
                self.events[p['event']]['central_start_s']=now
                p['first']=False
        result=np.clip(result,self.lower,self.upper)
        return result

    def ready(self,now,controller):
        return all(p['start'] is not None and now-p['start']>=self.duration for p in self.pending) and controller.get('central_mask',0)==self.mask
