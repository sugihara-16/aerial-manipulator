"""Private, instantaneous finite-contact response for bounded input allocation.

The only observations are position, velocity and the last issued input. This
is not a held-interval force guarantee; execution keeps its independent audit.
"""
import numpy as np
import mujoco
from scipy.linalg import cho_factor,cho_solve
from build.contact_tangent import contact_hessian, constraint_jacobian


def instantaneous_normal_floors(pressure, constant, minimum):
    """A zero unilateral response cannot supply positive instantaneous force.

    A merely weak contact is NOT removed. The restoration goal still uses
    minimum; these bounds apply only to the current instantaneous model.
    """
    absent=(np.linalg.norm(pressure,axis=1)<1e-12)&(np.abs(constant)<1e-12)
    return np.where(absent,0.,minimum)


class ContactAllocation:
    def __init__(self,local):
        self.local=local;self.m=local.m;self.d=mujoco.MjData(self.m)
        self.minimum_normal_load=.5
        self.anchor=None;self.object=self.m.geom('object').id
        self.pads={g:i for i,link in enumerate(local.links) for g in range(self.m.ngeom)
                   if self.m.geom(g).name.startswith(link+'/')}
        self.pad_index=np.full(self.m.ngeom,-1,dtype=int)
        for geom,pad in self.pads.items():self.pad_index[geom]=pad

    def linearize(self,q,v,mass,act,ids,scale,nominal):
        m,d,l=self.m,self.d,self.local
        d.qpos[:]=q;d.qvel[:]=v
        offset=l.kp*q[l.qi]+l.kd*v[l.vi]
        lower=np.r_[(offset-l.effort)/l.gain,l.rotor_min]
        upper=np.r_[(offset+l.effort)/l.gain,l.rotor_max]
        # Only optimized inputs have increment columns below. Fixed servo
        # inputs must already use this call's reference in the base response.
        base=nominal.copy()
        if self.anchor is not None:base[ids]=self.anchor[ids]
        base=np.clip(base,lower,upper)
        d.ctrl[:]=base;mujoco.mj_forward(m,d)
        if not d.efm_active:
            raise ValueError('contact allocation requires discrete effective dynamics')
        J=constraint_jacobian(m,d)
        C=contact_hessian(m,d,J@d.qacc-d.efc_aref)
        effective=mass+np.diag(d.efm_sdiag)+J.T@C@J
        response=cho_solve(cho_factor(effective,lower=True),act)
        shift=(nominal[ids]-base[ids])/scale[ids]
        acceleration=d.qacc+response@shift
        forces=-C@J@response;constant=d.efc_force+forces@shift
        pressure=np.zeros((len(l.links),len(ids)));pressure0=np.zeros(len(l.links))
        for c in d.contact:
            ga,gb=map(int,c.geom);adr=int(c.efc_address)
            if adr<0 or self.object not in (ga,gb):continue
            pad=self.pads.get(gb if ga==self.object else ga)
            if pad is not None:pressure[pad]+=forces[adr];pressure0[pad]+=constant[adr]
        self.instant_floor=instantaneous_normal_floors(pressure,pressure0,self.minimum_normal_load)
        return response,acceleration,pressure,pressure0

    def check(self,command,predicted_acceleration):
        m,d=self.m,self.d;d.ctrl[:]=command;mujoco.mj_forward(m,d)
        loads=np.zeros((len(self.local.links),3));moments=loads.copy();normal=np.zeros(len(loads))
        geometries=np.asarray(d.contact.geom)
        addresses=np.asarray(d.contact.efc_address)
        mask=(addresses>=0)&np.any(geometries==self.object,axis=1)
        indices=np.flatnonzero(mask)
        if len(indices):
            other=np.where(geometries[indices,0]==self.object,geometries[indices,1],geometries[indices,0])
            pad=self.pad_index[other];keep=pad>=0
            indices,other,pad=indices[keep],other[keep],pad[keep]
            force_local=d.efc_force[addresses[indices,None]+np.arange(3)]
            frames=np.asarray(d.contact.frame)[indices].reshape(-1,3,3)
            force=np.einsum('nij,ni->nj',frames,force_local)
            offset=np.asarray(d.contact.pos)[indices]-d.xpos[m.geom_bodyid[other]]
            np.add.at(normal,pad,force_local[:,0])
            np.add.at(loads,pad,force)
            np.add.at(moments,pad,np.cross(offset,force))
        self.anchor=command.copy()
        info=dict(acceleration_consistency=float(np.max(abs(d.qacc-predicted_acceleration))),
                  minimum_model_normal_n=float(min(normal)),maximum_model_force_n=float(max(np.linalg.norm(loads,axis=1))),
                  maximum_model_moment_nm=float(max(np.linalg.norm(moments,axis=1))))
        info['maximum_normal_deficit_n']=float(np.max(getattr(self,'instant_floor',self.minimum_normal_load)-normal))
        accepted=(info['maximum_normal_deficit_n']<=1e-7 and info['maximum_model_force_n']<=30.+1e-7
                  and info['maximum_model_moment_nm']<=5.+1e-7)
        return accepted,info
