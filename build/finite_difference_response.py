"""Input derivatives of the unchanged held-command contact response.

Each worker owns private prediction state, initialized only from observed q/v
and the proposed command. This never reads execution-side contact forces.
Finite differences include changes in contact geometry and friction branches;
they remain local approximations and require nonlinear constraint rechecking.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np,mujoco
from build.short_contact_response import ShortContactResponse


class FiniteDifferenceResponse(ShortContactResponse):
    def __init__(self,*args,workers=4,relative_step=1e-4,**kwargs):
        super().__init__(*args,**kwargs)
        self.pool=ThreadPoolExecutor(max_workers=workers);self.thread=threading.local()
        meta=self.plant.meta
        self.scale=np.r_[[a['effort'] for a in meta['actuators']],[r['maximum'] for r in meta['rotors']]]/self.m.actuator_gainprm[:,0]
        self.epsilon=relative_step*self.scale

    def close(self):
        self.pool.shutdown()

    def _response(self,args):
        q,v,u=args;m=self.m
        if not hasattr(self.thread,'data'):self.thread.data=mujoco.MjData(m)
        d=self.thread.data;mujoco.mj_resetData(m,d);d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=u
        states,loads,motors,penetration=self.runner.advance(d,self.steps,include_penetration=True)
        return dict(q=d.qpos.copy(),v=d.qvel.copy(),states=states,loads=loads,motors=motors,penetration=penetration)

    def evaluate(self,q,v,command,linearize=False):
        started=time.perf_counter();base=self._response((q,v,command));m=self.m
        if not linearize:return base
        def direction(i):
            plus=command.copy();minus=command.copy();plus[i]+=self.epsilon[i];minus[i]-=self.epsilon[i]
            pp=self._response((q,v,plus));mm=self._response((q,v,minus))
            D=np.empty((self.steps,2*m.nv))
            for k in range(self.steps):
                mujoco.mj_differentiatePos(m,D[k,:m.nv],2*self.epsilon[i],mm['states'][k,1:1+m.nq],pp['states'][k,1:1+m.nq])
            D[:,m.nv:]=(pp['states'][:,1+m.nq:]-mm['states'][:,1+m.nq:])/(2*self.epsilon[i])
            return D,(pp['loads']-mm['loads'])/(2*self.epsilon[i])
        values=list(self.pool.map(direction,range(m.nu)))
        base['derivatives']=np.stack([item[0]for item in values],axis=-1)
        base['load_derivatives']=np.stack([item[1]for item in values],axis=-1)
        base['linearization_s']=time.perf_counter()-started
        return base
