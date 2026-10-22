"""Reuse local input derivatives; keep every nonlinear interval check.

This is a chord iteration, not a frozen prediction of actual motion. Every
base/proposed input is propagated anew. Contact/actuator branch changes or
poor agreement for an input increment invalidate the derivative cache.
"""
import numpy as np,mujoco
from build.finite_difference_response import FiniteDifferenceResponse


class CachedResponse(FiniteDifferenceResponse):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cached=None;self.origin=None;self.signature=None
        self.counts=dict(refresh=0,reuse=0,inaccurate=0)

    def evaluate(self,q,v,command,linearize=False):
        base=self._response((q,v,command));d=self.thread.data;m=self.m
        signature=(tuple(np.asarray(d.contact.geom).ravel()),tuple(d.efc_state),
                   tuple(abs(base['motors'][:,0]-base['motors'][:,1]).max(axis=0)>1e-8))
        if linearize:
            if self.cached is None or signature!=self.signature:
                base=super().evaluate(q,v,command,True)
                self.cached=(base['derivatives'].copy(),base['load_derivatives'].copy())
                self.signature=signature;self.counts['refresh']+=1
            else:self.counts['reuse']+=1
            base['derivatives'],base['load_derivatives']=self.cached
            self.origin=(q.copy(),v.copy(),command.copy(),base)
        elif self.origin is not None and self.cached is not None:
            oq,ov,ou,previous=self.origin
            if np.array_equal(q,oq)and np.array_equal(v,ov):
                change=command-ou;D,L=self.cached
                expected_v=D[-1,m.nv:]@change;actual_v=base['v']-previous['v']
                expected_load=np.einsum('kpji,i->kpj',L,change);actual_load=base['loads']-previous['loads']
                # Relative agreement checks are numerical validity tests;
                # original nonlinear physical constraints are checked later.
                speed=np.full(m.nv,3.)
                for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
                    vi=m.jnt_dofadr[j];speed[vi:vi+6]=[.02]*3+[.1]*3
                numerator=max(np.linalg.norm((expected_v-actual_v)/speed),np.linalg.norm((expected_load-actual_load)/30.))
                denominator=max(np.linalg.norm(actual_v/speed),np.linalg.norm(actual_load/30.),1e-6)
                if numerator>.25*denominator:
                    self.cached=None;self.counts['inaccurate']+=1
        return base
