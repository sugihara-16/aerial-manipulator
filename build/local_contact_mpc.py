"""Short-horizon control of displacement, velocity and real held motor input.

The authored local compliant/sticking mechanics are linearized at one grasp.
Normal pressure remains tied to displacement and velocity. This is a local
model, not MuJoCo's exact reaction law or a global feasibility certificate.
"""
import time
import numpy as np
import scipy.sparse as sp
import clarabel,mujoco


class LocalContactMPC:
    def __init__(self,local,q,period=.01,horizon=5,fast_solve=False):
        began=time.perf_counter();self.local=local;self.q=q.copy();self.h=period;self.H=horizon;m=local.m
        u,report=local.equilibrium(q)
        if not report['feasible']:raise ValueError('local MPC requires a nominal equilibrium')
        self.u=u;self.gripper_tangents=len(report['tangent_force'])
        self.lam=np.r_[report['tangent_force'],np.asarray(report['support_force']).ravel()]
        self.feedback=local.linear_feedback(q,u)
        self.N=N=self.feedback['N'];self.n=n=N.shape[1];self.nu=nu=m.nu;self.nt=nt=len(self.lam)
        local.at(q);gap,_,Jn,Ct=local.contact();M,B,_=local.matrices();self.Jn=Jn;self.Ct=Ct;self.p=-local.k*gap
        reactions=np.vstack([Ct,local.support_matrix()])
        self.MN=M@N;self.B=B
        # Differentiate the complete generalized force, including changes in
        # thrust direction, normal/tangent geometry and the actuator servo.
        def force(pose,velocity):
            local.at(pose,velocity);g,_,jn,ct=local.contact();_,b,bias=local.matrices()
            reaction_map=np.vstack([ct,local.support_matrix()])
            return b@u+bias+jn.T@(-local.k*g-local.c*(jn@velocity))+reaction_map.T@self.lam
        K=np.empty((m.nv,n));D=K.copy();eps=1e-6
        for i in range(n):
            qp=q.copy();qm=q.copy();mujoco.mj_integratePos(m,qp,N[:,i],eps);mujoco.mj_integratePos(m,qm,N[:,i],-eps)
            K[:,i]=(force(qp,np.zeros(m.nv))-force(qm,np.zeros(m.nv)))/(2*eps)
            D[:,i]=(force(q,eps*N[:,i])-force(q,-eps*N[:,i]))/(2*eps)
        local.at(q);self.K=K;self.D=D;self.width=w=2*n+nu+nt
        self.scale=np.r_[np.full(n,.01),np.full(n,.1),local.effort/local.gain,local.rotor_max,np.ones(nt)]
        self.eq=np.zeros((n+m.nv,w));self.eq[:n,:n]=np.eye(n);self.eq[:n,n:2*n]=-period*np.eye(n)
        self.eq[n:,:n]=-K;self.eq[n:,n:2*n]=self.MN/period-D
        self.eq[n:,2*n:2*n+nu]=-B;self.eq[n:,2*n+nu:]=-reactions.T
        self.previous=np.zeros((n+m.nv,2*n));self.previous[:n,:n]=np.eye(n);self.previous[n:,n:]=self.MN/period
        pressure=np.zeros((len(gap),w));pressure[:,:n]=-local.k*Jn@N;pressure[:,n:2*n]=-local.c*Jn@N
        rows=[];bounds=[]
        def both(A,low,high):
            rows.extend(A);bounds.extend(np.broadcast_to(high,len(A)))
            rows.extend(-A);bounds.extend(-np.broadcast_to(low,len(A)))
        rows.extend(-pressure);bounds.extend(self.p)
        gap_map=np.zeros((len(gap),w));gap_map[:,:n]=Jn@N
        both(gap_map,-.002-gap,-gap)
        theta=np.arange(8)*np.pi/4;dirs=np.c_[np.cos(theta),np.sin(theta)];mu=4.5*np.cos(np.pi/8)
        for j in range(len(gap)):
            row=-mu*np.tile(pressure[j],(8,1));row[:,2*n+nu+2*j:2*n+nu+2*j+2]=dirs
            rows.extend(row);bounds.extend(np.full(8,mu*self.p[j])-dirs@self.lam[2*j:2*j+2])
        for j,support in enumerate(local.support):
            first=self.gripper_tangents+3*j;column=2*n+nu+first
            row=np.zeros(w);row[column]=-1.;rows.append(row);bounds.append(self.lam[first])
            friction=float(support['friction'])*np.cos(np.pi/8)
            polygon=np.zeros((8,w));polygon[:,column]=-friction;polygon[:,column+1:column+3]=dirs
            rows.extend(polygon);bounds.extend(friction*self.lam[first]-dirs@self.lam[first+1:first+3])
        effort=np.zeros((local.nj,w));effort[:,:n]=-local.kp[:,None]*N[local.vi];effort[:,n:2*n]=-local.kd[:,None]*N[local.vi]
        effort[:,2*n:2*n+local.nj]=np.diag(local.gain)
        effort0=local.gain*u[:local.nj]-local.kp*q[local.qi]
        both(effort,-local.effort-effort0,local.effort-effort0)
        rotor=np.zeros((nu-local.nj,w));rotor[:,2*n+local.nj:2*n+nu]=np.eye(nu-local.nj)
        both(rotor,local.rotor_min-u[local.nj:],local.rotor_max-u[local.nj:])
        position=np.zeros((local.nj,w));position[:,:n]=N[local.vi]
        both(position,m.jnt_range[local.joint_ids,0]-q[local.qi],m.jnt_range[local.joint_ids,1]-q[local.qi])
        velocity=np.zeros((local.nj,w));velocity[:,n:2*n]=N[local.vi];both(velocity,-3.,3.)
        gimbals=np.flatnonzero(~local.dock);command=np.zeros((len(gimbals),w));command[np.arange(len(gimbals)),2*n+gimbals]=1.
        both(command,m.jnt_range[local.joint_ids[gimbals],0]-u[gimbals],m.jnt_range[local.joint_ids[gimbals],1]-u[gimbals])
        # Local relative-pose region. Common translation is exact and removed.
        relative=np.eye(m.nv);obj=m.jnt_dofadr[m.joint('object').id];trust=np.full(m.nv,.005)
        for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
            vi=m.jnt_dofadr[j];relative[vi:vi+3,obj:obj+3]-=np.eye(3);trust[vi:vi+3]=.001
        region=np.zeros((m.nv,w));region[:,:n]=relative@N;both(region,-trust,trust)
        pads=np.array([e[0]for e in local.elements]);self.maps=[];self.constants=[]
        for pad in range(len(local.links)):
            selected=np.flatnonzero(pads==pad);rows.append(-pressure[selected].sum(axis=0));bounds.append(self.p[selected].sum()-.5)
            F=np.zeros((3,w));T=F.copy();f0=np.zeros(3);t0=f0.copy()
            for j in selected:
                _,bp,bo,lp,_,lf=local.elements[j];frame=lf@local.d.xmat[bo].reshape(3,3).T;arm=local.d.xmat[bp].reshape(3,3)@lp
                action=frame[0,:,None]*pressure[j];action[:,2*n+nu+2*j:2*n+nu+2*j+2]+=frame[1:].T
                constant=frame.T@np.r_[self.p[j],self.lam[2*j:2*j+2]]
                skew=np.array([[0,-arm[2],arm[1]],[arm[2],0,-arm[0]],[-arm[1],arm[0],0]])
                F+=action;T+=skew@action;f0+=constant;t0+=skew@constant
            for matrix,value,limit in [(F,f0,30.),(T,t0,5.)]:
                self.maps.append(matrix);self.constants.append(value)
                # Each physical-pad norm is retained as an exact cone.
        bounds=np.asarray(bounds);finite=np.isfinite(bounds);self.ineq=np.asarray(rows)[finite];self.bound=bounds[finite]
        block=np.zeros((w,w));block[:2*n,:2*n]=self.feedback['Q']*period
        block[2*n:2*n+nu,2*n:2*n+nu]=self.feedback['R']*period
        block[2*n+nu:,2*n+nu:]=np.eye(nt)*1e-9
        blocks=[block.copy()for _ in range(horizon)];blocks[-1][:2*n,:2*n]+=self.feedback['P']
        self.cost=sp.block_diag(blocks,format='csc');self.scale_all=np.tile(self.scale,horizon)
        self.cost=sp.diags(self.scale_all)@self.cost@sp.diags(self.scale_all)
        eqblocks=sp.kron(sp.eye(horizon),sp.csc_matrix(self.eq))
        prev=np.c_[self.previous,np.zeros((len(self.previous),nu+nt))]
        eqblocks-=sp.kron(sp.diags(np.ones(horizon-1),-1,shape=(horizon,horizon)),sp.csc_matrix(prev))
        inequality=sp.kron(sp.eye(horizon),sp.csc_matrix(self.ineq))
        cone_matrix=[];cone_bounds=[]
        for k in range(horizon):
            for index,(matrix,value)in enumerate(zip(self.maps,self.constants)):
                row=sp.lil_matrix((4,horizon*w));row[1:,k*w:(k+1)*w]=-matrix
                cone_matrix.append(row.tocsc());cone_bounds.extend([30. if index%2==0 else 5.,*value])
        self.equal_count=eqblocks.shape[0];self.matrix=sp.vstack([eqblocks,inequality,*cone_matrix],format='csc')@sp.diags(self.scale_all)
        self.bound_all=np.r_[np.zeros(self.equal_count),np.tile(self.bound,horizon),cone_bounds]
        self.cones=[clarabel.ZeroConeT(self.equal_count),clarabel.NonnegativeConeT(len(self.bound)*horizon)]+[clarabel.SecondOrderConeT(4)for _ in cone_matrix]
        self.settings=clarabel.DefaultSettings();self.settings.verbose=False;self.settings.max_threads=1;self.settings.direct_solve_method='qdldl'
        self.settings.tol_gap_abs=self.settings.tol_gap_rel=self.settings.tol_feas=1e-8
        self.linear=np.zeros(horizon*w);self.solver=None;self.qp=None;self.fast_solve=fast_solve;self.preparation_s=time.perf_counter()-began

    def update(self,q,v,references,input_reference=None):
        began=time.perf_counter();m=self.local.m;n,w=self.n,self.width;delta=np.zeros(m.nv)
        mujoco.mj_differentiatePos(m,delta,1.,self.q,q);state=np.r_[self.N.T@delta,self.N.T@v]
        self.bound_all[:len(self.previous)]=self.previous@state
        self.linear[:]=0.
        for k,(pose,velocity)in enumerate(references):
            mujoco.mj_differentiatePos(m,delta,1.,self.q,pose);target=np.r_[self.N.T@delta,self.N.T@velocity]
            weight=self.feedback['Q']*self.h+(self.feedback['P']if k==self.H-1 else 0.)
            self.linear[k*w:k*w+2*n]=-weight@target
            if input_reference is not None:
                self.linear[k*w+2*n:k*w+2*n+self.nu]=-self.h*self.feedback['R']@(input_reference-self.u)
        linear=self.linear*self.scale_all
        from build.held_input_solver import violation
        z=None;fallback=False;status=None;iterations=None
        if self.fast_solve:
            import proxsuite
            ne=self.equal_count;ni=self.cones[1].dim
            if self.qp is None:
                self.qp=proxsuite.proxqp.sparse.QP(len(linear),ne,ni)
                self.qp.settings.eps_abs=self.qp.settings.eps_rel=1e-9
                self.qp.settings.max_iter=100;self.qp.settings.max_iter_in=100
                self.qp.init(self.cost.tocsc(),linear,self.matrix[:ne].tocsc(),self.bound_all[:ne],
                    self.matrix[ne:ne+ni].tocsc(),np.full(ni,-np.inf),self.bound_all[ne:ne+ni])
            else:
                self.qp.settings.initial_guess=proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                self.qp.update(g=linear,b=self.bound_all[:ne],u=self.bound_all[ne:ne+ni])
            self.qp.solve();candidate=self.qp.results.x
            residual=violation(self.matrix,self.bound_all,self.cones,candidate)
            status=str(self.qp.results.info.status);iterations=int(self.qp.results.info.iter)
            if status.endswith('PROXQP_SOLVED')and np.isfinite(residual)and residual<=1e-5:z=candidate.copy()
            else:fallback=True
        if z is None:
            if self.solver is None:self.solver=clarabel.DefaultSolver(sp.triu(self.cost,format='csc'),linear,self.matrix,self.bound_all,self.cones,self.settings)
            else:self.solver.update(q=linear,b=self.bound_all)
            answer=self.solver.solve();status=str(answer.status);iterations=int(answer.iterations)
            if status not in ['Solved','AlmostSolved']:raise ValueError('local MPC '+status)
            z=np.asarray(answer.x);residual=violation(self.matrix,self.bound_all,self.cones,z)
        if residual>1e-5:raise ValueError('local MPC constraint residual '+str(residual))
        x=(z*self.scale_all).reshape(self.H,w);first=x[0]
        self.predicted_q=self.q.copy();mujoco.mj_integratePos(m,self.predicted_q,self.N@first[:n],1.)
        self.predicted_v=self.N@first[n:2*n]
        pressure=self.p-self.local.k*(self.Jn@self.N@first[:n])-self.local.c*(self.Jn@self.predicted_v)
        self.predicted_loads=np.bincount([e[0]for e in self.local.elements],weights=pressure,minlength=len(self.local.links))
        self.predicted_support_forces=(self.lam+first[2*n+self.nu:])[self.gripper_tangents:].reshape(-1,3)
        self.diagnostics=dict(status=status,fallback=fallback,constraint_error=residual,iterations=iterations,
                              update_s=time.perf_counter()-began,model_position=first[:n].tolist(),model_velocity=first[n:2*n].tolist())
        return self.u+first[2*n:2*n+self.nu]
