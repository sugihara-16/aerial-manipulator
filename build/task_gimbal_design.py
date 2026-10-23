"""Continuous fixed-gimbal design for hold and task acceleration endpoints.

This small nonlinear problem constrains external wrench only. Its result MUST
pass the existing finite-contact, drive and clearance checks before adoption.
No trajectory is simulated to choose the angles.
"""
import time
from functools import lru_cache
import numpy as np
import mujoco



@lru_cache(maxsize=32)
def _gimbal_solver(influenced,ns):
    import casadi as ca
    ng=len(influenced);nr=sum(map(len,influenced));width=ng+ns*nr
    x=ca.SX.sym('x',width);parameters=[]
    def parameter(name,r,c=1):
        p=ca.SX.sym(name,r,c);parameters.append(ca.vec(p));return p
    angles=parameter('angles',ng);axes=parameter('axes',3,ng)
    anchors=parameter('anchors',3,ng);forces=parameter('forces',3,nr);arms=parameter('arms',3,nr)
    moving_mass=parameter('moving_mass',3,ng);mass_arm=parameter('mass_arm',3)
    specific=parameter('specific',3,ns);corrections=parameter('corrections',6,ns)
    mass=parameter('mass',1);fmax=parameter('fmax',nr);scale=parameter('scale',6);initial=parameter('initial',width)
    columns=[None]*nr
    for j,ids in enumerate(influenced):
        axis=axes[:,j];skew=ca.skew(axis);angle=x[j]-angles[j]
        R=ca.DM.eye(3)+ca.sin(angle)*skew+(1-ca.cos(angle))*(skew@skew)
        for i in ids:
            force=R@forces[:,i];arm=anchors[:,j]+R@(arms[:,i]-anchors[:,j])
            columns[i]=ca.vertcat(force,ca.cross(arm,force))
        mass_arm=mass_arm+(R-ca.DM.eye(3))@moving_mass[:,j]
    A=ca.horzcat(*columns);con=[]
    for k in range(ns):
        target=ca.vertcat(mass*specific[:,k],ca.cross(mass_arm,specific[:,k]))+corrections[:,k]
        con.append((A@(x[ng+k*nr:ng+(k+1)*nr]*fmax)-target)/scale)
    weights=np.r_[np.ones(ng),np.full(ns*nr,1e-8)]
    obj=.5*ca.dot(weights*(x-initial),x-initial)
    return ca.nlpsol('repair','ipopt',dict(x=x,p=ca.vertcat(*parameters),f=obj,g=ca.vertcat(*con)),
        {'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.max_iter':80,
         'ipopt.tol':1e-10,'ipopt.constr_viol_tol':1e-10,'ipopt.bound_relax_factor':0.})


class GimbalWrenchDesign:
    def __init__(self,local,q,displacement,corrections=None):
        self.local,self.q=local,q.copy();self.m=local.m;self.d=mujoco.MjData(self.m)
        self.gimbals=np.flatnonzero(~local.dock)
        self.joints=local.joint_ids[self.gimbals];self.qi=local.qi[self.gimbals]
        self.nr=len(local.sites);self.ng=len(self.gimbals)
        self.mass=float(self.m.body_mass.sum());self.oi=int(self.m.jnt_qposadr[self.m.joint('object').id])
        lift=np.array([0.,0.,.15])*10/np.sqrt(3)/4.**2
        carry=np.asarray(displacement)*10/np.sqrt(3)/6.**2
        self.accelerations=np.array([np.zeros(3),lift,-lift,carry,-carry])
        self.corrections=corrections
        if corrections is not None:self.accelerations=np.zeros((len(corrections),3))
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
        if self.corrections is not None:targets+=self.corrections
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
        initial=np.clip(np.r_[angles,np.tile(u[self.local.nj:]/fmax,ns)],low,high)
        A0,_,_,_=self.geometry(angles);origin=self.q[self.oi:self.oi+3]
        span=max(float(np.linalg.norm(self.d.site_xpos[self.local.sites]-origin,axis=1).max()),.01)
        scale=np.r_[np.full(3,self.mass*9.80665),np.full(3,self.mass*9.80665*span)]
        covered=[i for ids in self.influenced for i in ids]
        if sorted(covered)!=list(range(nr)):raise ValueError('symbolic design requires independent rotor hinges')
        axes=self.d.xaxis[self.joints].T;anchors=(self.d.xanchor[self.joints]-origin).T
        bodies=self.m.jnt_bodyid[self.joints]
        moving_mass=(self.m.body_subtreemass[bodies,None]*(self.d.subtree_com[bodies]-self.d.xanchor[self.joints])).T
        values=[angles,axes,anchors,A0[:3],(self.d.site_xpos[self.local.sites]-origin).T,
            moving_mass,np.sum(self.m.body_mass[:,None]*(self.d.xipos-origin),axis=0),
            (self.accelerations-self.m.opt.gravity).T,
            np.zeros((6,ns)) if self.corrections is None else self.corrections.T,
            np.array([self.mass]),fmax,scale,initial]
        parameters=np.concatenate([np.asarray(v).ravel(order='F') for v in values])
        solver=_gimbal_solver(tuple(tuple(ids) for ids in self.influenced),ns)
        answer=solver(x0=initial,p=parameters,lbx=low,ubx=high,lbg=0,ubg=0);result=np.asarray(answer['x']).ravel()
        A,b,_,_=self.geometry(result[:ng]);forces=result[ng:].reshape(ns,nr)*fmax
        residual=float(abs(forces@A.T-b).max());pose=self.q.copy();pose[self.qi]=result[:ng]
        accepted=bool(np.isfinite(residual)and residual<1e-6 and np.all(result>=low-1e-9)and np.all(result<=high+1e-9))
        return pose,dict(scope=__doc__,accepted=accepted,solver_status=solver.stats()['return_status'],iterations=solver.stats()['iter_count'],
            external_wrench_residual=residual,retained_rotor_fraction=reserve,maximum_angle_change_rad=float(abs(result[:ng]-angles).max()),wall_s=time.perf_counter()-began)
