"""Local empty-robot feedback from authored inertia and actual actuator gains.

The supported object's coordinates are excluded. There is no contact force
input or simulation rollout in synthesis. Stability is local to the nominal
free-flight equilibrium; actual release transitions are evaluated separately.
"""
import numpy as np
from build.sticking_design import LocalGrasp
from build.unloaded_hover import command


class UnloadedFlight(LocalGrasp):
    def __init__(self,plant,q):
        super().__init__(plant,q,[])
        index=int(self.m.jnt_dofadr[self.m.joint('object').id])
        self.object_constraint=np.eye(self.m.nv)[index:index+6]

    def support_matrix(self):
        return self.object_constraint

    def equilibrium(self,q,acceleration=None):
        previous=np.r_[np.where(self.dock,self.kp*q[self.qi],q[self.qi]),np.zeros(len(self.sites))]
        u,info=command(self,q,previous,acceleration)
        return u,dict(info,feasible=True)
