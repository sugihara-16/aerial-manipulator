"""A common translation reference driven by attainable path acceleration."""
import numpy as np
import mujoco
from amsrr_det.path_progress import ScalarProgress


class TranslationGovernor(ScalarProgress):
    def __init__(self,model,start,displacement,duration,period=.01):
        delta=np.asarray(displacement);distance=float(np.linalg.norm(delta))
        super().__init__(distance,duration,period)
        self.m,self.start=model,start.copy();self.direction=delta/distance
        self.direction6=np.r_[self.direction,np.zeros(3)]
        self.free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        self.oi=int(model.jnt_qposadr[model.joint('object').id]);self.ov=int(model.jnt_dofadr[model.joint('object').id])

    def reference(self):
        q=self.start.copy();v=np.zeros(self.m.nv)
        for j in self.free:
            qi=self.m.jnt_qposadr[j];vi=self.m.jnt_dofadr[j]
            q[qi:qi+3]+=self.position*self.direction;v[vi:vi+3]=self.speed*self.direction
        return q,v

    def observed_choice(self,interval,q,v,feedback):
        return self.choose(interval,
            observed_position=float((q[self.oi:self.oi+3]-self.start[self.oi:self.oi+3])@self.direction),
            observed_speed=float(v[self.ov:self.ov+3]@self.direction),
            feedback_acceleration=float(feedback[:3]@self.direction))
