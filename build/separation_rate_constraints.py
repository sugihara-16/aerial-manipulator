"""Observed environment separation rates for the acquisition input solve.

These are local velocity constraints, not a continuous collision certificate.
The independent full-geometry audit still checks every executed physics step.
No contact force or successful simulated trajectory enters these constraints.
"""
import numpy as np
import mujoco
from amsrr_det.clearance import MotionBoundedClearance


class SeparationRateConstraints:
    def __init__(self,model,links,response_time=.5):
        self.query=MotionBoundedClearance(model,links,halo=.02)
        self.response_time=response_time
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

    def at(self,q,v):
        m,d=self.query.model,self.query.data
        d.qpos[:]=q;d.qvel[:]=v
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
        intended,limits=self.query.classifiers[True](d)
        rows=[];lower=[]
        for c,allowed,limit in zip(d.contact,intended,limits):
            if allowed:continue
            a,b=map(int,c.geom)
            ja=np.zeros((3,m.nv));jb=ja.copy()
            mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[a]))
            mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[b]))
            rows.append(c.frame[:3]@(jb-ja))
            lower.append(-(c.dist-limit)/self.response_time)
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
