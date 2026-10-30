"""Observed environment separation rates for the acquisition input solve.

These are local velocity constraints, not a continuous collision certificate.
The independent full-geometry audit still checks every executed physics step.
No contact force or successful simulated trajectory enters these constraints.
"""
import copy
import numpy as np
import mujoco
from amsrr_det.clearance import MotionBoundedClearance, collision_query


class SeparationRateConstraints:
    def __init__(self,model,links,response_time=.5,*,all_geometry=False,distance_tolerance=0.,distance_query=None):
        self._task_priority=False;self.private_query=distance_query is None
        if distance_query is None:
            self.query=MotionBoundedClearance(model,links,halo=.02,always_allow_selected=True)
        else:
            if not all_geometry:
                raise ValueError('shared distance model requires all geometry')
            # Match the acceptance query's collision geometry and buffers.
            # Different detection halos can change curved-shape CCD roundoff
            # enough to strand an otherwise valid boundary correction.
            self.query=copy.copy(distance_query)
            self.query.data=mujoco.MjData(distance_query.model)
        self.response_time=response_time
        self.distance_tolerance=distance_tolerance
        if all_geometry:
            return  # Preserve the full model's collision pairs and intended-contact exclusions.
        # This query protects robot/environment distances. Internal geometry
        # remains in the independent full-geometry execution audit; it need
        # not enter this additional environment velocity constraint.
        m=self.query.model;m.geom_contype[:]=0;m.geom_conaffinity[:]=0
        for i in range(m.ngeom):
            name=m.geom(i).name
            if name in ['floor','support','destination_support']:m.geom_contype[i]=1
            elif name.startswith('m')and '/'in name:m.geom_conaffinity[i]=1
        m.body_contype[:]=0;m.body_conaffinity[:]=0
        np.bitwise_or.at(m.body_contype,m.geom_bodyid,m.geom_contype)
        np.bitwise_or.at(m.body_conaffinity,m.geom_bodyid,m.geom_conaffinity)
        mujoco.mj_setConst(m,mujoco.MjData(m))

    @property
    def task_priority(self):return self._task_priority

    @task_priority.setter
    def task_priority(self,value):
        self._task_priority=bool(value)
        if self.private_query:
            # This halo is a cache horizon, not a required physical gap.
            # The task-priority path only needs observed non-penetration;
            # asking CCD for every pair within an extra 2 cm dominates time.
            halo=.003 if value else .02
            query=self.query;query.model.geom_gap+=(halo-query.halo)/2
            query.halo=halo;query.anchor=None
            mujoco.mj_setConst(query.model,mujoco.MjData(query.model))

    def at(self,q,v):
        if getattr(self,'task_priority',False):
            # Reuse the geometric proof of a non-colliding OBSERVED pose, not
            # the extra 1 mm / 3 cm buffers. This is not a future-motion proof;
            # actual collision remains audited at every physical step.
            query=self.query
            if query.anchor is not None and query.joint_motion is not None:
                oldq,_,_,part,environment=query.anchor
                movement=2.*float(query.joint_motion.maximum(q,oldq)[0])
                internal=2.*float(query.joint_motion.internal_maximum(q,oldq)[0])
                if part>internal and environment>movement:
                    query.reused+=1
                    return np.empty((0,query.model.nv)),np.empty(0)
            _,part,environment=query.check(q,True)
            if min(part,environment)>0.:
                return np.empty((0,query.model.nv)),np.empty(0)
        m,d=self.query.model,self.query.data
        d.qpos[:]=q;d.qvel[:]=v
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);collision_query(m,d)
        intended,limits=self.query.classifiers[True](d)
        rows=[];lower=[]
        for c,allowed,limit in zip(d.contact,intended,limits):
            if allowed:continue
            a,b=map(int,c.geom)
            ja=np.zeros((3,m.nv));jb=ja.copy()
            mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[a]))
            mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[b]))
            rows.append(c.frame[:3]@(jb-ja))
            lower.append(-(c.dist-limit+self.distance_tolerance)/self.response_time)
        return np.asarray(rows).reshape(-1,m.nv),np.asarray(lower)


def preload_in_physical_domain(model,data,links):
    """Authored normal compliance and conservative Coulomb pad bounds.

This replaces a numerical closeness-to-target test with the existing physical
domain. It is a model check, not a reading of the simulator's contact forces.
"""
    normal=np.zeros(len(links));moment=normal.copy();object_id=model.geom('object').id
    factor=np.sqrt(1+4.5**2)
    for c in data.contact:
        a,b=map(int,c.geom)
        if object_id not in (a,b)or c.dist>0:continue
        other=b if a==object_id else a;name=model.geom(other).name
        pad=next((i for i,link in enumerate(links)if name.startswith(link+'/')),None)
        if pad is None:continue
        if c.dist<-.002:return False
        load=-7500.*c.dist;normal[pad]+=load
        moment[pad]+=factor*load*np.linalg.norm(c.pos-data.xpos[model.geom_bodyid[other]])
    return bool(np.all(normal>=.5)and np.all(factor*normal<=30.)and np.all(moment<=5.))


class PreloadDomain:
    """Guide aggregate pad compression into the existing holding-entry domain."""
    def __init__(self,clearance,model,links,period):
        self.clearance,self.m,self.links,self.period=clearance,model,links,period
        self.d=mujoco.MjData(model)

    def at(self,q,v):
        rows,bounds=self.clearance.at(q,v)
        self.d.qpos[:]=q;self.d.qvel[:]=v;mujoco.mj_fwdPosition(self.m,self.d)
        extra,limits=preload_domain_rates(self.m,self.d,self.links,self.period)
        return np.vstack([rows,extra]),np.r_[bounds,limits]


def preload_domain_rates(m,d,links,period,*,maintain_floor=False):
    # Recover the same conservative preload domain required at handoff.
    # Guide excessive compression down over the existing 0.5 s response
    # time; do not demand an instantaneous force drop from a bounded motor.
    normal=np.zeros(len(links));derivative=np.zeros((len(links),m.nv));lever=normal.copy()
    obj=m.geom('object').id
    for c in d.contact:
        ga,gb=map(int,c.geom)
        if obj not in (ga,gb) or c.dist>0.:continue
        other=gb if ga==obj else ga
        pad=next((i for i,link in enumerate(links)if m.geom(other).name.startswith(link+'/')),None)
        if pad is None:continue
        ja=np.zeros((3,m.nv));jb=ja.copy()
        mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[ga]))
        mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[gb]))
        derivative[pad]+=c.frame[:3]@(jb-ja)
        normal[pad]+=-7500.*c.dist
        lever[pad]=max(lever[pad],np.linalg.norm(c.pos-d.xpos[int(m.geom_bodyid[other])]))
    limit=np.minimum(30./np.sqrt(1+4.5**2),5./(np.sqrt(1+4.5**2)*np.maximum(lever,1e-12)))
    # Retain 5% interior room, as in the existing finite preload design.
    target=.95*limit
    rate=np.where(normal>target,(normal-target)/(7500.*.5),(normal-target)/(7500.*period))
    if maintain_floor:
        # Recover toward an interior normal load, using the same 5% room as
        # the upper bound. Only established contacts have this local Jacobian.
        present=normal>0.
        floor=.5/.95
        lower=(floor-normal[present])/(7500.*.5)
        return np.vstack([derivative,-derivative[present]]),np.r_[rate,lower]
    return derivative,rate
