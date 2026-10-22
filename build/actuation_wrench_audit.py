"""Audit requested total actuator wrench against independent rotor-site sums.

Internal joint torques cancel in the whole assembly. No contact-force signal
enters the controller; this class only records issuance and held-period error.
"""
import numpy as np,mujoco


class ActuationWrenchAudit:
    def __init__(self,plant):
        self.m=plant.mj;self.d=mujoco.MjData(self.m);self.nj=len(plant.meta['actuators'])
        self.sites=[self.m.site(plant.pin.frames[r['frame']].name).id for r in plant.meta['rotors']]
        self.peak_issue=np.zeros(6);self.peak_hold=np.zeros(6);self.impulse=np.zeros(6)
        self.request=None;self.commands=0;self.steps=0

    def wrench(self,data,thrust):
        force=data.site_xmat[self.sites].reshape(-1,3,3)[:,:,2]*thrust[:,None]
        return np.r_[force.sum(axis=0),np.cross(data.site_xpos[self.sites],force).sum(axis=0)]

    def issue(self,q,command,request):
        self.d.qpos[:]=q;mujoco.mj_kinematics(self.m,self.d)
        self.request=np.asarray(request).copy()
        limits=self.m.actuator_forcerange[self.nj:]
        raw=self.m.actuator_gainprm[self.nj:,0]*command[self.nj:]
        thrust=np.where(self.m.actuator_forcelimited[self.nj:],np.clip(raw,limits[:,0],limits[:,1]),raw)
        self.peak_issue=np.maximum(self.peak_issue,abs(self.wrench(self.d,thrust)-self.request));self.commands+=1

    def after_step(self,data):
        if self.request is None:return
        error=self.wrench(data,data.actuator_force[self.nj:])-self.request
        self.peak_hold=np.maximum(self.peak_hold,abs(error));self.impulse+=self.m.opt.timestep*error;self.steps+=1

    def result(self):
        return dict(scope=__doc__,commands=self.commands,steps=self.steps,
            peak_issuance_difference=self.peak_issue.tolist(),peak_held_difference=self.peak_hold.tolist(),
            accumulated_difference_impulse=self.impulse.tolist(),units='force N, moment N m; integrated N s, N m s')
