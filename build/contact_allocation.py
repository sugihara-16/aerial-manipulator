"""Private, instantaneous finite-contact response for bounded input allocation.

The only observations are position, velocity and the last issued input. This
is not a held-interval force guarantee; execution keeps its independent audit.
"""
import numpy as np
import mujoco
from scipy.linalg import cho_factor,cho_solve
from build.contact_tangent import contact_hessian, constraint_jacobian


class ContactAllocation:
    def __init__(self,local):
        self.local=local;self.m=local.m;self.d=mujoco.MjData(self.m)
        self.minimum_normal_load=.5
        self.anchor=None;self.object=self.m.geom('object').id
        self.pads={g:i for i,link in enumerate(local.links) for g in range(self.m.ngeom)
                   if self.m.geom(g).name.startswith(link+'/')}

    def linearize(self,q,v,mass,act,ids,scale,nominal):
        m,d,l=self.m,self.d,self.local
        d.qpos[:]=q;d.qvel[:]=v
        offset=l.kp*q[l.qi]+l.kd*v[l.vi]
        lower=np.r_[(offset-l.effort)/l.gain,l.rotor_min]
        upper=np.r_[(offset+l.effort)/l.gain,l.rotor_max]
        base=np.clip(nominal if self.anchor is None else self.anchor,lower,upper)
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
        return response,acceleration,pressure,pressure0

    def check(self,command,predicted_acceleration):
        m,d=self.m,self.d;d.ctrl[:]=command;mujoco.mj_forward(m,d)
        loads=np.zeros((len(self.local.links),3));moments=loads.copy();normal=np.zeros(len(loads))
        for c in d.contact:
            ga,gb=map(int,c.geom);adr=int(c.efc_address)
            if adr<0 or self.object not in (ga,gb):continue
            other=gb if ga==self.object else ga;pad=self.pads.get(other)
            if pad is None:continue
            normal[pad]+=d.efc_force[adr]
            force=c.frame.reshape(3,3).T@d.efc_force[adr:adr+3]
            loads[pad]+=force;moments[pad]+=np.cross(c.pos-d.xpos[int(m.geom_bodyid[other])],force)
        self.anchor=command.copy()
        info=dict(acceleration_consistency=float(np.max(abs(d.qacc-predicted_acceleration))),
                  minimum_model_normal_n=float(min(normal)),maximum_model_force_n=float(max(np.linalg.norm(loads,axis=1))),
                  maximum_model_moment_nm=float(max(np.linalg.norm(moments,axis=1))))
        accepted=(info['minimum_model_normal_n']>=self.minimum_normal_load-1e-7 and info['maximum_model_force_n']<=30.+1e-7
                  and info['maximum_model_moment_nm']<=5.+1e-7)
        return accepted,info
