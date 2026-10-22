"""Common observed-state tracking through the complete held-input response."""
import numpy as np,mujoco
from build.response_contact_control import ResponseContactControl
from build.cached_response import CachedResponse
from build.contact_step_control import pad_gap_jacobians


class IntervalTrackingControl(ResponseContactControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,iterations=2,fast_solve=True,adaptive_damping=True,**kwargs)
        self.predictor=CachedResponse(self.plant,self.links)
        self.dock=~self.gimbal

    def reset_search(self):
        """Restart only numerical approximations at a new reference segment.

        The last issued input and all observed physical state are retained.
        """
        self.predictor.cached=None
        self.predictor.origin=None
        self.predictor.signature=None
        self.damping=1e-3

    def update(self,q,v,*args,**kwargs):
        command=super().update(q,v,*args,**kwargs)
        m,d=self.m,self.query;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_fwdPosition(m,d)
        obj=m.geom('object').id;speed=0.
        for c in d.contact:
            a,b=map(int,c.geom)
            if obj not in (a,b):continue
            other=b if a==obj else a
            if self.predictor.runner.groups[other]<0:continue
            ja=np.zeros((3,m.nv));jb=ja.copy()
            mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[a]))
            mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[b]))
            speed=max(speed,float(np.linalg.norm((c.frame.reshape(3,3)@(jb-ja)@v)[1:])))
        d.qpos[:]=self.predicted_q;mujoco.mj_fwdPosition(m,d)
        self.predicted_gaps=pad_gap_jacobians(m,d,self.links,self.groups)[0]
        self.diagnostics['observed_tangent_speed']=speed
        return command
