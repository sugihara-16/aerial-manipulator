"""Continuous fixed-gimbal design for hold and task acceleration endpoints.

This small nonlinear problem constrains external wrench only. Its result MUST
pass the existing finite-contact, drive and clearance checks before adoption.
No trajectory is simulated to choose the angles.
"""
import time
import numpy as np
import mujoco
from scipy.optimize import minimize,Bounds


class GimbalWrenchDesign:
    def __init__(self,local,q,displacement):
        self.local,self.q=local,q.copy();self.m=local.m;self.d=mujoco.MjData(self.m)
        self.gimbals=np.flatnonzero(~local.dock)
        self.joints=local.joint_ids[self.gimbals];self.qi=local.qi[self.gimbals]
        self.nr=len(local.sites);self.ng=len(self.gimbals)
        self.mass=float(self.m.body_mass.sum());self.oi=int(self.m.jnt_qposadr[self.m.joint('object').id])
        lift=np.array([0.,0.,.15])*10/np.sqrt(3)/4.**2
        carry=np.asarray(displacement)*10/np.sqrt(3)/6.**2
        self.accelerations=np.array([np.zeros(3),lift,-lift,carry,-carry])
        self.influenced=[]
        for joint in self.joints:
            body=int(self.m.jnt_bodyid[joint]);indices=[]
            for i,site in enumerate(local.sites):
                ancestor=int(self.m.site_bodyid[site])
                while ancestor and ancestor!=body:ancestor=int(self.m.body_parentid[ancestor])
                if ancestor==body:indices.append(i)
            self.influenced.append(indices)

    def geometry(self,angles):
        m,d=self.m,self.d;d.qpos[:]=self.q;d.qpos[self.qi]=angles
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        origin=d.qpos[self.oi:self.oi+3]
        force=d.site_xmat[self.local.sites].reshape(-1,3,3)[:,:,2]
        arm=d.site_xpos[self.local.sites]-origin
        matrix=np.r_[force.T,np.cross(arm,force).T]
        mass_arm=np.sum(m.body_mass[:,None]*(d.xipos-origin),axis=0)
        specific=self.accelerations-m.opt.gravity
        targets=np.c_[self.mass*specific,np.cross(mass_arm,specific)]
        derivative=np.zeros((self.ng,6,self.nr));target_derivative=np.zeros((self.ng,len(targets),6))
        for j,(joint,indices) in enumerate(zip(self.joints,self.influenced)):
            axis=d.xaxis[joint];anchor=d.xanchor[joint];ids=np.array(indices,dtype=int)
            dz=np.cross(axis,force[ids]);dp=np.cross(axis,d.site_xpos[self.local.sites][ids]-anchor)
            derivative[j,:3,ids]=dz
            derivative[j,3:,ids]=np.cross(dp,force[ids])+np.cross(arm[ids],dz)
            body=int(m.jnt_bodyid[joint])
            dm=m.body_subtreemass[body]*np.cross(axis,d.subtree_com[body]-anchor)
            target_derivative[j,:,3:]=np.cross(dm,specific)
        return matrix,targets,derivative,target_derivative

    def solve(self,u,reserve=0.):
        began=time.perf_counter();ng,nr=self.ng,self.nr;ns=len(self.accelerations)
        angles=self.q[self.qi];fmax=self.local.rotor_max
        low=np.r_[self.m.jnt_range[self.joints,0],np.tile(self.local.rotor_min/fmax+reserve,ns)]
        high=np.r_[self.m.jnt_range[self.joints,1],np.full(ns*nr,1-reserve)]
        initial=np.r_[angles,np.tile(u[self.local.nj:]/fmax,ns)]
        initial=np.clip(initial,low,high)
        self.geometry(angles)
        span=max(float(np.linalg.norm(self.d.site_xpos[self.local.sites]-self.q[self.oi:self.oi+3],axis=1).max()),.01)
        scale=np.r_[np.full(3,self.mass*9.80665),np.full(3,self.mass*9.80665*span)]
        def constraints(x):
            A,b,da,db=self.geometry(x[:ng]);forces=x[ng:].reshape(ns,nr)*fmax
            values=(forces@A.T-b)/scale
            jac=np.zeros((ns,6,len(x)))
            jac[:,:,:ng]=(np.einsum('jki,si->skj',da,forces)-db.transpose(1,2,0))/scale[None,:,None]
            for s in range(ns):jac[s,:,ng+s*nr:ng+(s+1)*nr]=A*fmax/scale[:,None]
            return values.ravel(),jac.reshape(-1,len(x))
        def objective(x):
            # Repair the current feasible contact posture with the least angle
            # change. The later finite-input solve and structure comparison use
            # the actual thrust burden; this is not a global angle optimum.
            weights=np.r_[np.ones(ng),np.full(ns*nr,1e-8)]
            delta=x-initial
            return .5*float(np.sum(weights*delta**2)),weights*delta
        answer=minimize(objective,initial,jac=True,method='SLSQP',bounds=Bounds(low,high),
            constraints=[dict(type='eq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],
            options=dict(maxiter=80,ftol=1e-10))
        A,b,_,_=self.geometry(answer.x[:ng]);forces=answer.x[ng:].reshape(ns,nr)*fmax
        residual=float(abs(forces@A.T-b).max());pose=self.q.copy();pose[self.qi]=answer.x[:ng]
        accepted=bool(np.isfinite(residual) and residual<1e-6 and np.all(answer.x>=low-1e-9) and np.all(answer.x<=high+1e-9))
        return pose,dict(scope=__doc__,accepted=accepted,solver_status=str(answer.message),iterations=answer.nit,
            external_wrench_residual=residual,retained_rotor_fraction=reserve,
            maximum_angle_change_rad=float(abs(answer.x[:ng]-angles).max()),wall_s=time.perf_counter()-began)
