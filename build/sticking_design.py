"""Experimental local mechanics: normal compliance, ideal material sticking.

Only geometry, authored inertia and actuator parameters are read from MuJoCo.
No simulation stepping, force truth, or trajectory-based candidate acceptance.
The finite contact elements are fixed for ONE local model. Mode changes and
agreement with independent dynamics require separate validation.
"""
import numpy as np
import mujoco
from scipy.linalg import null_space,expm,solve_discrete_are
from scipy.optimize import minimize,LinearConstraint
from amsrr_det.linear_program import linprog


class LocalGrasp:
    def __init__(self,plant,q,links,stiffness=7500.,damping=75.,object_shape=None,support=(),constraint_rcond=1e-9):
        self.plant=plant;self.m=plant.mj;self.d=mujoco.MjData(self.m)
        self.k=stiffness;self.c=damping;self.elements=[];self.links=links
        self.support=list(support)
        self.constraint_rcond=constraint_rcond
        self.at(q);oid=self.m.geom('object').id
        for contact in self.d.contact:
            ga,gb=map(int,contact.geom)
            if oid not in (ga,gb):continue
            other=gb if ga==oid else ga
            pads=[i for i,s in enumerate(links) if self.m.geom(other).name.startswith(s+'/')]
            if not pads or contact.dist>1e-7:continue
            bp=int(self.m.geom_bodyid[other]);bo=int(self.m.geom_bodyid[oid])
            frame=contact.frame.reshape(3,3).copy()*(1 if gb==oid else -1)
            point=contact.pos.copy();normal=frame[0]
            pp=point-.5*contact.dist*normal;po=point+.5*contact.dist*normal
            Rp=self.d.xmat[bp].reshape(3,3);Ro=self.d.xmat[bo].reshape(3,3)
            if object_shape is not None:
                # Known object geometry provides surface normals without CCD
                # iterative normal noise. Keep the pad material point.
                projected,_,outward=object_shape.project_surface(Ro.T@(pp-self.d.xpos[bo]))
                po=self.d.xpos[bo]+Ro@projected[0];normal=-Ro@outward[0]
                tangent=frame[1]-normal*(normal@frame[1]);tangent/=np.linalg.norm(tangent)
                frame=np.array([normal,tangent,np.cross(normal,tangent)])
            self.elements.append((pads[0],bp,bo,Rp.T@(pp-self.d.xpos[bp]),
                                  Ro.T@(po-self.d.xpos[bo]),frame@Ro))
        self.joint_ids=np.array([self.m.joint(a['name']).id for a in plant.meta['actuators']])
        self.qi=self.m.jnt_qposadr[self.joint_ids];self.vi=self.m.jnt_dofadr[self.joint_ids]
        self.nj=len(self.vi)
        self.kp=np.array([a['kp'] for a in plant.meta['actuators']])
        self.kd=np.array([a['kd'] for a in plant.meta['actuators']])
        self.effort=np.array([a['effort'] for a in plant.meta['actuators']])
        self.dock=np.array([a['role']=='dock' for a in plant.meta['actuators']])
        self.gain=np.where(self.dock,1.,self.kp)
        self.sites=[self.m.site(self.plant.pin.frames[r['frame']].name).id for r in plant.meta['rotors']]
        self.rotor_max=np.array([r['maximum'] for r in plant.meta['rotors']])
        self.rotor_min=np.array([r['minimum'] for r in plant.meta['rotors']])

    def at(self,q,v=None):
        self.d.qpos[:]=q;self.d.qvel[:]=0 if v is None else v
        mujoco.mj_fwdPosition(self.m,self.d);mujoco.mj_fwdVelocity(self.m,self.d)

    def contact(self):
        gap=[];tangent=[];Jn=[];Ct=[]
        for pad,bp,bo,lp,lo,local_frame in self.elements:
            pp=self.d.xpos[bp]+self.d.xmat[bp].reshape(3,3)@lp
            po=self.d.xpos[bo]+self.d.xmat[bo].reshape(3,3)@lo
            frame=local_frame@self.d.xmat[bo].reshape(3,3).T
            jp=np.zeros((3,self.m.nv));jo=jp.copy();rot=jp.copy()
            mujoco.mj_jac(self.m,self.d,jp,None,pp,bp)
            mujoco.mj_jac(self.m,self.d,jo,rot,po,bo)
            delta=po-pp
            derivative=frame@(jo-jp)+np.cross(frame,delta)@rot
            values=frame@delta
            gap.append(values[0]);tangent.extend(values[1:]);Jn.append(derivative[0]);Ct.extend(derivative[1:])
        return np.array(gap),np.array(tangent),np.array(Jn).reshape(-1,self.m.nv),np.array(Ct).reshape(-1,self.m.nv)

    def support_matrix(self):
        from build.support_contacts import support_jacobian
        return support_jacobian(self.m,self.d,self.support)

    def sticking_constraints(self,Ct):
        return np.vstack([Ct,self.support_matrix()])

    def matrices(self):
        M=np.zeros((self.m.nv,self.m.nv));mujoco.mj_fullM(self.m,self.d,M)
        B=np.zeros((self.m.nv,self.nj+len(self.sites)))
        B[self.vi,np.arange(self.nj)]=self.gain
        for i,site in enumerate(self.sites):
            J=np.zeros((3,self.m.nv));mujoco.mj_jacSite(self.m,self.d,J,None,site)
            B[:,self.nj+i]=J.T@self.d.site_xmat[site].reshape(3,3)[:,2]
        bias=-self.d.qfrc_bias.copy()
        # MIT dock input is the held combined command; local gains persist.
        bias[self.vi]-=self.kp*self.d.qpos[self.qi]+self.kd*self.d.qvel[self.vi]
        return M,B,bias

    def acceleration(self,q,v,u):
        self.at(q,v);g,_,Jn,Ct=self.contact();M,B,bias=self.matrices()
        p=self.k*(-g)-self.c*(Jn@v)
        F=B@u+bias+Jn.T@p
        # Local linearization at rest only: Cdot*v is second order there.
        # Use the same independent velocity subspace as feedback synthesis.
        # Squaring C in a normal equation otherwise loses weak constraints.
        constraints=self.sticking_constraints(Ct)
        N=null_space(constraints,rcond=self.constraint_rcond)
        acceleration=N@np.linalg.solve(N.T@M@N,N.T@F)
        reaction=np.linalg.lstsq(constraints.T,M@acceleration-F,rcond=self.constraint_rcond)[0]
        return acceleration,p,reaction

    def close_by(self,q,depth):
        """Diagnostic compression, not yet an optimized preload generator."""
        q=q.copy()
        for _ in range(6):
            self.at(q);g,t,Jn,Ct=self.contact()
            depths=np.broadcast_to(depth,(len(self.links),))
            target=np.array([depths[e[0]] for e in self.elements])
            residual=np.r_[-target-g,-t]
            if np.max(abs(residual))<1e-10:break
            J=np.vstack([Jn,Ct]);obj=self.m.jnt_dofadr[self.m.joint('object').id]
            J[:,obj:obj+6]=0
            delta=np.linalg.lstsq(J,residual,rcond=1e-10)[0]
            mujoco.mj_integratePos(self.m,q,delta,1.)
        return q

    def static_burden(self,q):
        """Minimum simultaneous normalized motor/friction use at fixed pose.

        Existence screen only: an indeterminate tangent distribution is NOT a
        prediction that the drive realizes that distribution. Preload search
        uses this to construct a diagnostic equilibrium for separate testing.
        """
        self.at(q);g,t,Jn,Ct=self.contact();M,B,bias=self.matrices();p=-self.k*g
        if not len(p) or min(p)<=0:return np.inf,None
        nj=self.nj;nu=B.shape[1];nt=len(Ct);width=nu+nt+1
        offset=np.r_[np.where(self.dock,self.kp*q[self.qi],q[self.qi]),np.zeros(len(self.sites))]
        eq=np.c_[B,Ct.T,np.zeros(self.m.nv)];rhs=-bias-Jn.T@p-B@offset
        rows=[];limits=[]
        for j in range(nj):
            row=np.zeros(width);row[j]=self.gain[j];row[-1]=-self.effort[j]
            rows.extend([row.copy(),-row]);rows[-1][-1]=-self.effort[j];limits.extend([0,0])
        theta=np.arange(8)*2*np.pi/8
        for j in range(len(p)):
            for angle in theta:
                row=np.zeros(width);row[nu+2*j:nu+2*j+2]=[np.cos(angle),np.sin(angle)]
                row[-1]=-4.5*p[j]*np.cos(np.pi/8);rows.append(row);limits.append(0.)
        bounds=[(None,None)]*nj+list(zip(self.rotor_min,self.rotor_max))+[(None,None)]*nt+[(0,None)]
        objective=np.zeros(width);objective[-1]=1
        result=linprog(objective,A_eq=eq,b_eq=rhs,A_ub=np.array(rows),b_ub=np.array(limits),bounds=bounds)
        if not result.success:return np.inf,None
        return float(result.x[-1]),offset+result.x[:nu]

    def equilibrium(self,q,acceleration=None,*,finite_slip_duration=None):
        self.at(q);g,t,Jn,Ct=self.contact();M,B,bias=self.matrices();p=-self.k*g
        required=np.zeros(self.m.nv) if acceleration is None else M@np.asarray(acceleration)
        support=self.support_matrix();ns=len(support);nu=B.shape[1];nt=len(Ct)
        target=required-bias-Jn.T@p;A=np.column_stack([B,Ct.T,support.T])
        reaction_velocity=None
        if finite_slip_duration is not None:
            if ns or acceleration is not None or finite_slip_duration<=0:
                raise ValueError('finite-slip diagnostic requires an airborne quasistatic point and positive duration')
            from build.momentum_feedback import rigid_motion_map
            joints=np.zeros((self.nj,self.m.nv));joints[np.arange(self.nj),self.vi]=1.
            # A local steady slip may move material points, but must not
            # accumulate joint deflection or change the normal compression.
            # Remove common momentum to separate it from transport motion.
            allowed=null_space(np.vstack([Jn,joints,rigid_motion_map(self.m,q).T@M]),rcond=self.constraint_rcond)
            slip=Ct@allowed
            incompatible=null_space(slip.T,rcond=self.constraint_rcond).T
            rows=np.zeros((len(incompatible),A.shape[1]));rows[:,nu:nu+nt]=incompatible
            A=np.vstack([A,rows]);target=np.r_[target,np.zeros(len(rows))]
            reaction_velocity=-allowed@np.linalg.pinv(slip,rcond=self.constraint_rcond)/(self.c*self.m.opt.impratio)
        nominal=np.r_[np.where(self.dock,self.kp*q[self.qi],q[self.qi]),np.zeros(len(self.sites))]
        support_scale=max(float(self.plant.meta['object']['mass'])*9.80665/max(len(self.support),1),.01)
        scale=np.r_[self.effort/self.gain,self.rotor_max,np.repeat(np.maximum(p,.01),2),np.full(ns,support_scale)]
        offset=np.r_[nominal,np.zeros(nt+ns)]
        low=np.r_[nominal[:self.nj]-self.effort/self.gain,self.rotor_min,
                  np.full(nt,-np.inf),np.tile([0.,-np.inf,-np.inf],len(self.support))]
        high=np.r_[nominal[:self.nj]+self.effort/self.gain,self.rotor_max,
                   np.full(nt+ns,np.inf)]
        # Inscribed polygon: this diagnostic uses a sufficient Coulomb bound.
        friction=np.zeros((8*(len(p)+len(self.support)),len(scale)))
        for i in range(len(p)):
            theta=np.arange(8)*2*np.pi/8
            friction[8*i:8*(i+1),B.shape[1]+2*i:B.shape[1]+2*i+2]=np.c_[np.cos(theta),np.sin(theta)]
        for i,item in enumerate(self.support):
            row=8*(len(p)+i);col=nu+nt+3*i
            theta=np.arange(8)*2*np.pi/8
            friction[row:row+8,col]=-item['friction']*np.cos(np.pi/8)
            friction[row:row+8,col+1:col+3]=np.c_[np.cos(theta),np.sin(theta)]
        force_limit=np.r_[np.repeat(4.5*p*np.cos(np.pi/8),8),np.zeros(8*len(self.support))]
        if reaction_velocity is not None:
            # These finite-time bounds are on a local constant-rate model,
            # not a promise that real geometry stays fixed for the interval.
            rate=np.full(self.m.nv,3.)
            for joint in np.flatnonzero(self.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
                vi=int(self.m.jnt_dofadr[joint]);rate[vi:vi+6]=np.r_[[.03]*3,[.2]*3]/(finite_slip_duration*np.sqrt(3))
            travel=np.vstack([reaction_velocity/rate[:,None],
                Ct@reaction_velocity*(finite_slip_duration*np.sqrt(2)/.03)])
            extra=np.zeros((len(travel),len(scale)));extra[:,nu:nu+nt]=travel
            friction=np.vstack([friction,extra,-extra]);force_limit=np.r_[force_limit,np.ones(2*len(extra))]
        scaled=A*scale;rhs=target-A@offset
        # Independent equilibrium equations avoid redundant SLSQP equalities.
        U,s,V=np.linalg.svd(scaled,full_matrices=False);rank=np.count_nonzero(s>s[0]*1e-10)
        equality=(U[:,:rank].T@scaled)/s[:rank,None]
        value=(U[:,:rank].T@rhs)/s[:rank]
        lower=(low-offset)/scale;upper=(high-offset)/scale
        width=len(scale);actuated=B.shape[1]
        # A local unsaturated controller needs an interior operating point.
        # Maximize a COMMON fraction of rated input/friction reserve first.
        upper_rows=np.zeros((actuated,width+1));upper_rows[:,:actuated]=np.eye(actuated);upper_rows[:,-1]=1.
        lower_rows=upper_rows.copy();lower_rows[:,:actuated]*=-1
        margin_rows=np.vstack([upper_rows,lower_rows,np.c_[friction*scale,force_limit]])
        margin_limits=np.r_[upper[:actuated],-lower[:actuated],force_limit-friction@offset]
        linear=linprog(np.r_[np.zeros(width),-1.],A_eq=np.c_[scaled,np.zeros(len(rhs))],b_eq=rhs,
                       A_ub=margin_rows,b_ub=margin_limits,
                       bounds=list(zip(lower,upper))+[(0.,.5)],method='highs')
        if not linear.success:
            return None,dict(feasible=False,linear_status=int(linear.status),message=linear.message,
                             normal_force=p.tolist(),equilibrium_singular_values=s.tolist())
        reserve=float(linear.x[-1]);retained=.95*reserve
        lower[:actuated]+=retained;upper[:actuated]-=retained
        result=minimize(lambda x:.5*x@x,linear.x[:-1],jac=lambda x:x,method='SLSQP',
            bounds=list(zip(lower,upper)),
            constraints=[LinearConstraint(equality,value,value),
                         LinearConstraint(friction*scale,-np.inf,force_limit*(1-retained)-friction@offset)],
            options=dict(ftol=1e-11,maxiter=100))
        z=offset+scale*result.x;u=z[:nu];reaction=z[nu:nu+nt]
        residual=float(abs(A@z-target).max())
        slip_info={}
        if reaction_velocity is not None:
            slip_velocity=reaction_velocity@reaction
            slip_info=dict(finite_slip_duration_s=finite_slip_duration,nominal_slip_velocity=slip_velocity.tolist(),
                tangent_law_residual=float(abs(reaction+self.c*self.m.opt.impratio*(Ct@slip_velocity)).max(initial=0.)),
                normal_rate_max=float(abs(Jn@slip_velocity).max(initial=0.)),joint_rate_max=float(abs(slip_velocity[self.vi]).max(initial=0.)))
        return u,dict(**slip_info,solver_success=bool(result.success),message=result.message,iterations=int(result.nit),
                      balance_error=residual,normal_force=p.tolist(),tangent_force=reaction.tolist(),
                      support_force=z[nu+nt:].reshape(-1,3).tolist(),
                      minimum_gap=float(g.min(initial=np.inf)),material_error=float(abs(t).max(initial=0.)),
                      maximum_common_reserve=reserve,retained_common_reserve=retained,
                      feasible=bool(result.success and residual<1e-7 and p.min(initial=np.inf)>0))

    def linear_feedback(self,q,u,period=.01,epsilon=1e-6):
        self.at(q);_,_,_,Ct=self.contact();N=null_space(self.sticking_constraints(Ct),rcond=self.constraint_rcond);n=N.shape[1]
        A=np.zeros((2*n,2*n));A[:n,n:]=np.eye(n);B=np.zeros((2*n,len(u)))
        for i in range(n):
            delta=N[:,i]*epsilon;qp=q.copy();qm=q.copy()
            mujoco.mj_integratePos(self.m,qp,delta,1);mujoco.mj_integratePos(self.m,qm,delta,-1)
            A[n:,i]=N.T@(self.acceleration(qp,np.zeros(self.m.nv),u)[0]-self.acceleration(qm,np.zeros(self.m.nv),u)[0])/(2*epsilon)
            A[n:,n+i]=N.T@(self.acceleration(q,delta,u)[0]-self.acceleration(q,-delta,u)[0])/(2*epsilon)
        self.at(q);_,_,_,Ct=self.contact();M,act,_=self.matrices()
        B[n:]=np.linalg.solve(N.T@M@N,N.T@act)
        block=np.zeros((2*n+len(u),2*n+len(u)));block[:2*n,:2*n]=A;block[:2*n,2*n:]=B
        sampled=expm(block*period);Ad=sampled[:2*n,:2*n];Bd=sampled[:2*n,2*n:]
        ps=np.ones(self.m.nv);vs=np.ones(self.m.nv)
        for j in range(self.m.njnt):
            vi=self.m.jnt_dofadr[j]
            if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
                ps[vi:vi+6]=[.03,.03,.03,.2,.2,.2];vs[vi:vi+6]=[.02,.02,.02,.1,.1,.1]
            else:ps[vi]=max(np.diff(self.m.jnt_range[j])[0]/2,.01);vs[vi]=3.
        Q=np.zeros_like(A);Q[:n,:n]=N.T@np.diag(ps**-2)@N;Q[n:,n:]=N.T@np.diag(vs**-2)@N
        us=np.r_[self.effort/self.gain,self.rotor_max];R=np.diag(us**-2)
        P=solve_discrete_are(Ad,Bd,Q,R)
        K=np.linalg.solve(R+Bd.T@P@Bd,Bd.T@P@Ad)
        eig=np.linalg.eigvals(Ad-Bd@K)
        return dict(N=N,A=A,B=B,Ad=Ad,Bd=Bd,Q=Q,R=R,P=P,K=K,eigenvalues=eig)
