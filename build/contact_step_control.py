"""Experimental inverse one-period dynamics with finite contact forces.

The decision is next velocity plus actual held motor input. Existing contacts
may stick at the END of the period only if the same finite friction/drive
constraints permit it. Regularized pad friction includes boundary branches
when the interior law is infeasible. Geometry uses a private q/v model.
"""
import numpy as np
import mujoco
import scipy.sparse as sp
import clarabel


def pad_gap_jacobians(m,d,links,groups=None):
    """Closest physical surface gap and its instantaneous relative Jacobian."""
    obj=m.geom('object').id;gaps=[];rows=[]
    if groups is None:groups=[tuple(g for g in range(m.ngeom) if m.geom(g).name.startswith(link+'/'))for link in links]
    for geoms in groups:
        geom_set=set(geoms);active=[]
        for c in d.contact:
            ga,gb=map(int,c.geom)
            if (ga==obj and gb in geom_set) or (gb==obj and ga in geom_set):active.append(c)
        if active:
            c=min(active,key=lambda c:c.dist);a,b=map(int,c.geom);pa=pb=c.pos
            normal=c.frame[:3];gap=c.dist
        else:
            gap=np.inf;segment=np.zeros(6)
            for a in geoms:
                # A farther surface cannot improve the closest-distance
                # result. Tighten the query bound after each closer witness.
                distance=mujoco.mj_geomDistance(m,d,a,obj,min(1.,gap)if gap>0 else 1.,segment)
                if distance<gap:gap=distance;nearest=segment.copy();closest=a
            if not np.isfinite(gap) or gap>=1.:raise ValueError('pad surface is outside geometry query')
            a,b=closest,obj;pa,pb=nearest[:3],nearest[3:]
            normal=(pb-pa)/np.linalg.norm(pb-pa)
        ja=np.zeros((3,m.nv));jb=ja.copy()
        mujoco.mj_jac(m,d,ja,None,pa,int(m.geom_bodyid[a]))
        mujoco.mj_jac(m,d,jb,None,pb,int(m.geom_bodyid[b]))
        gaps.append(gap);rows.append(normal@(jb-ja))
    return np.array(gaps),np.array(rows)


class ContactStepControl:
    def __init__(self,plant,links,period=.01,input_weight=1e-6,regularized_friction=False,finite_support=False,fast_solve=False,distance_closing=False,condensed_solve=False,clearance_model=None):
        self.separation=None
        if clearance_model is not None:
            from build.separation_rate_constraints import SeparationRateConstraints
            self.separation=SeparationRateConstraints(clearance_model,links)
        self.plant=plant;self.m=plant.mj;self.d=mujoco.MjData(self.m);self.links=links;self.h=period
        self.groups=[tuple(g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith(link+'/'))for link in links]
        self.input_weight=input_weight
        self.regularized_friction=regularized_friction
        self.finite_support=finite_support
        self.fast_solve=fast_solve
        self.distance_closing=distance_closing
        self.condensed_solve=condensed_solve
        self.object_velocity_bounds=None
        self.support_load_upper=None
        self.minimum_normal_load=0.
        self.support_force_priority=False
        self.fixed_input=None
        self.project_input=False
        self.project_object=False
        self.friction_reserve=0.
        meta=plant.meta;self.nj=len(meta['actuators']);self.nu=self.m.nu;self.nv=self.m.nv
        self.ids=np.array([self.m.joint(a['name']).id for a in meta['actuators']])
        self.qi=self.m.jnt_qposadr[self.ids];self.vi=self.m.jnt_dofadr[self.ids]
        self.kp=np.array([a['kp'] for a in meta['actuators']]);self.kd=np.array([a['kd'] for a in meta['actuators']])
        self.effort=np.array([a['effort'] for a in meta['actuators']]);self.dock=np.array([a['role']=='dock' for a in meta['actuators']])
        self.qlo=np.array([a['lower'] for a in meta['actuators']]);self.qhi=np.array([a['upper'] for a in meta['actuators']])
        self.sites=[self.m.site(plant.pin.frames[r['frame']].name).id for r in meta['rotors']]
        self.limit=np.r_[self.effort,[r['maximum'] for r in meta['rotors']]]
        self.speed=np.full(self.nv,3.)
        for j in np.flatnonzero(self.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
            v=self.m.jnt_dofadr[j];self.speed[v:v+6]=[.02]*3+[.1]*3

    def update(self,q,v,reference,velocity=None,input_reference=None,normal_targets=None):
        m,d,h=self.m,self.d,self.h;nv,nu,nj=self.nv,self.nu,self.nj
        d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_fwdPosition(m,d);mujoco.mj_fwdVelocity(m,d)
        M=np.zeros((nv,nv));mujoco.mj_fullM(m,d,M)
        B=np.zeros((nv,nu));B[self.vi,np.arange(nj)]=1.
        for i,site in enumerate(self.sites):
            J=np.zeros((3,nv));mujoco.mj_jacSite(m,d,J,None,site);B[:,nj+i]=J.T@d.site_xmat[site].reshape(3,3)[:,2]
        normals=[];gaps=[];tangents=[];supports=[];mus=[];contact_geometry=[];normal_regularization=[];support_response=[]
        obj=m.geom('object').id
        for c in d.contact:
            ga,gb=map(int,c.geom)
            if obj not in (ga,gb) or c.dist>1e-9:continue
            other=gb if ga==obj else ga;name=m.geom(other).name
            pad=next((i for i,link in enumerate(self.links) if name.startswith(link+'/')),None)
            selected=pad is not None
            supported=name in ['support','destination_support','floor']
            if not selected and not supported:continue
            ja=np.zeros((3,nv));jb=ja.copy()
            mujoco.mj_jac(m,d,ja,None,c.pos,int(m.geom_bodyid[ga]))
            mujoco.mj_jac(m,d,jb,None,c.pos,int(m.geom_bodyid[gb]))
            J=c.frame.reshape(3,3)@(jb-ja)
            if selected:
                normals.append(J[0]);tangents.extend(J[1:]);gaps.append(c.dist)
                contact_geometry.append((pad,c.frame.reshape(3,3).copy()*(1. if ga==obj else -1.),
                                         c.pos-d.xpos[int(m.geom_bodyid[other])]))
                normal_regularization.append(99.*sum(m.body_invweight0[m.geom_bodyid[g],0] for g in (ga,gb)))
            else:
                supports.extend(J);mus.append(float(max(m.geom_friction[ga,0],m.geom_friction[gb,0])))
                if self.finite_support:
                    if c.solimp[0]!=c.solimp[1]:raise ValueError('finite support requires the authored constant impedance')
                    impedance=float(c.solimp[0]);maximum=float(c.solimp[1]);time_constant,ratio=map(float,c.solref)
                    if time_constant>0:
                        stiffness=1./(maximum**2*time_constant**2*ratio**2);damping=2./(maximum*time_constant)
                    else:stiffness=-time_constant/maximum**2;damping=-ratio/maximum
                    rn=(1-impedance)/impedance*sum(m.body_invweight0[m.geom_bodyid[g],0] for g in (ga,gb))
                    rt=rn/m.opt.impratio;kn=impedance*stiffness/rn;cn=damping/rn
                    coefficients=np.array([h*kn+cn+1/(h*rn),damping/rt+1/(h*rt),damping/rt+1/(h*rt)])
                    rest=(J@v)/(h*np.array([rn,rt,rt]));rest[0]-=kn*c.dist
                    support_response.append((1./coefficients,rest/coefficients))
        Jn=np.array(normals).reshape(-1,nv);Ct=np.array(tangents).reshape(-1,nv);Js=np.array(supports).reshape(-1,nv)
        ne=len(Jn);nt=len(Ct);ns=len(Js);size=nv+nu+nt+ns
        # With no robot/object contact, the object/support dynamics are an
        # independent subsystem. Robot inputs cannot stop its rolling or
        # rebound. Track the observed object at constant velocity for this
        # 10 ms geometric target, without imposing a support-sticking solve.
        # This is not a prediction/certificate of the object's contact forces.
        free_object=not ne
        if free_object:
            Js=np.zeros((0,nv));ns=0;mus=[];support_response=[];size=nv+nu
        # End-of-period normal force p = p0 - Dn*v+. Actual drive effort
        # is e + Kd*v - (Kd+h*Kp)*v+, where e is its initial effort.
        p0=-7500.*np.array(gaps);Dn=(75.+h*7500.)*Jn
        if self.regularized_friction:
            rn=np.array(normal_regularization)
            p0+=(Jn@v)/(h*rn);Dn+=Jn/(h*rn[:,None])
        E=M/h+Jn.T@Dn
        E[self.vi,self.vi]+=self.kd+h*self.kp
        rhs=M@v/h-d.qfrc_bias+Jn.T@p0
        rhs[self.vi]+=self.kd*v[self.vi]
        if free_object:
            oi=int(m.jnt_dofadr[m.joint('object').id]);indices=np.arange(oi,oi+6)
            other=np.setdiff1d(np.arange(nv),indices)
            if np.max(abs(M[np.ix_(indices,other)]),initial=0.)>1e-12 or np.max(abs(B[indices]),initial=0.)>1e-12:
                raise ValueError('uncontacted object is not dynamically independent')
            E[indices]=0.;E[indices,indices]=1.;rhs[indices]=v[indices]
        rows=[np.c_[E,-B,-Ct.T,-Js.T]];low=[rhs];high=[rhs]
        # Coefficients of a normal-pressure correction, used only when the
        # coupled friction boundary law replaces the interior contact law.
        pressure_blocks={0:-Jn.T}
        if self.object_velocity_bounds is not None and not free_object:
            start=int(m.jnt_dofadr[m.joint('object').id])
            object_velocity=np.zeros((6,size));object_velocity[np.arange(6),start+np.arange(6)]=1.
            rows.append(object_velocity)
            low.append(np.asarray(self.object_velocity_bounds[0]));high.append(np.asarray(self.object_velocity_bounds[1]))
        if self.separation is not None:
            separation,bound=self.separation.at(q,v)
            rows.append(np.c_[separation,np.zeros((len(separation),size-nv))])
            low.append(bound);high.append(np.full(len(separation),np.inf))
        C=np.vstack([Ct,Js]);contact_rows=np.c_[C,np.zeros((len(C),size-nv))];contact_rhs=np.zeros(len(C))
        if self.regularized_friction:
            # Numerical friction compliance implied by the same coating
            # conversion and MuJoCo impratio; this is not a material spring.
            rt=np.repeat(rn/m.opt.impratio,2);dt=75.*m.opt.impratio
            contact_rows[np.arange(nt),nv+nu+np.arange(nt)]=1./(1./(h*rt)+dt)
            contact_rhs[:nt]=(Ct@v)/(1.+h*rt*dt)
        if self.finite_support and ns:
            diagonal=np.concatenate([x[0] for x in support_response]);value=np.concatenate([x[1] for x in support_response])
            contact_rows[nt+np.arange(ns),nv+nu+nt+np.arange(ns)]=diagonal;contact_rhs[nt:]=value
        rows.append(contact_rows);low.append(contact_rhs);high.append(contact_rhs)
        # Nonnegative normal pressure and unchanged physical motor limits.
        rows.append(np.c_[Dn,np.zeros((ne,size-nv))]);low.append(np.full(ne,-np.inf));high.append(p0)
        pressure_blocks[len(rows)-1]=-np.eye(ne)
        rows.append(np.c_[h*Jn,np.zeros((ne,size-nv))]);low.append(-.002-np.array(gaps));high.append(np.full(ne,np.inf))
        if np.any(np.asarray(self.minimum_normal_load)>0):
            pad_pressure=np.zeros((len(self.links),size));pad_constant=np.zeros(len(self.links))
            for i,(pad,_,_) in enumerate(contact_geometry):
                pad_pressure[pad,:nv]+=Dn[i];pad_constant[pad]+=p0[i]
            rows.append(pad_pressure);low.append(np.full(len(self.links),-np.inf))
            high.append(pad_constant-self.minimum_normal_load)
            membership=np.zeros((len(self.links),ne))
            for i,(pad,_,_) in enumerate(contact_geometry):membership[pad,i]=-1.
            pressure_blocks[len(rows)-1]=membership
        if self.support_load_upper is not None and ns:
            if not self.finite_support:raise ValueError('support unloading requires finite support response')
            supported_load=np.zeros((1,size));supported_load[0,nv+nu+nt+np.arange(0,ns,3)]=1.
            rows.append(supported_load);low.append(np.array([-np.inf]));high.append(np.array([self.support_load_upper]))
        select=np.zeros((nu,size));select[:,nv:nv+nu]=np.eye(nu)
        elo=-self.limit.copy();elo[nj:]=0.;ehi=self.limit.copy()
        # Gimbal remains a bounded position command; Dock uses MIT fields.
        elo[:nj]=np.maximum(elo[:nj],np.where(self.dock,-np.inf,self.kp*(self.qlo-q[self.qi])-self.kd*v[self.vi]))
        ehi[:nj]=np.minimum(ehi[:nj],np.where(self.dock,np.inf,self.kp*(self.qhi-q[self.qi])-self.kd*v[self.vi]))
        rows.append(select);low.append(elo);high.append(ehi)
        end=select[:nj].copy();end[np.arange(nj),self.vi]=-(self.kd+h*self.kp)
        rows.append(end);low.append(-self.effort-self.kd*v[self.vi]);high.append(self.effort-self.kd*v[self.vi])
        speed=np.zeros((nj,size));speed[np.arange(nj),self.vi]=1.
        rows.append(speed);low.append(np.maximum(-3.,(self.qlo-q[self.qi])/h));high.append(np.minimum(3.,(self.qhi-q[self.qi])/h))
        theta=np.arange(8)*np.pi/4;directions=np.c_[np.cos(theta),np.sin(theta)]
        friction=np.zeros((8*(ne+len(mus))+len(mus),size));flimit=np.zeros(len(friction))
        for i in range(ne):
            friction[8*i:8*i+8,:nv]=4.5*np.cos(np.pi/8)*Dn[i]
            friction[8*i:8*i+8,nv+nu+2*i:nv+nu+2*i+2]=directions
            flimit[8*i:8*i+8]=4.5*np.cos(np.pi/8)*p0[i]
            if self.friction_reserve>0:
                pad=contact_geometry[i][0]
                count=sum(item[0]==pad for item in contact_geometry)
                flimit[8*i:8*i+8]-=4.5*np.cos(np.pi/8)*.5*self.friction_reserve/count
        for i,mu in enumerate(mus):
            r=8*(ne+i);col=nv+nu+nt+3*i
            friction[r:r+8,col]=-mu*np.cos(np.pi/8);friction[r:r+8,col+1:col+3]=directions
            friction[8*(ne+len(mus))+i,col]=-1.
        rows.append(friction);low.append(np.full(len(friction),-np.inf));high.append(flimit)
        pressure_friction=np.zeros((len(friction),ne))
        for i in range(ne):pressure_friction[8*i:8*i+8,i]=-4.5*np.cos(np.pi/8)
        pressure_blocks[len(rows)-1]=pressure_friction
        delta=np.zeros(nv);mujoco.mj_differentiatePos(m,delta,1.,q,reference)
        goal=(np.zeros(nv) if velocity is None else velocity)+delta/.5
        weights=np.r_[self.speed**-2,self.input_weight/self.limit**2,np.full(nt+ns,1e-9)]
        target=np.r_[goal,np.zeros(nu+nt+ns)]
        if input_reference is not None:
            target[nv:nv+nj]=np.where(self.dock,input_reference[:nj],self.kp*input_reference[:nj])-self.kp*q[self.qi]-self.kd*v[self.vi]
            target[nv+nj:nv+nu]=input_reference[nj:]
        objective=np.diag(weights);linear=-weights*target
        pad_gaps=None
        if normal_targets is not None:
            pad_gaps,jacobian=pad_gap_jacobians(m,d,self.links,self.groups)
            scale=.0002
            # Away from contact, use the same 0.5 s geometric convergence
            # time instead of imposing the contact acquisition speed over
            # millimetres of empty space. Existing input/velocity/load bounds
            # remain part of the same solve; near contact the old cap remains.
            cap=np.maximum(scale,np.maximum(pad_gaps,0.)/.5)if self.distance_closing else scale
            rate=np.clip((-np.asarray(normal_targets)-pad_gaps)/.5,-cap,scale)
            task=np.zeros((len(rate),size));task[:,:nv]=jacobian/scale
            objective+=task.T@task;linear-=task.T@(rate/scale)
        # The force and moment limits concern each complete physical pad,
        # not each collision element separately. Use exact norm cones.
        force=np.zeros((len(self.links),3,size));constant=np.zeros((len(self.links),3))
        moment=force.copy();moment_constant=constant.copy()
        pressure_force=np.zeros((len(self.links),3,ne));pressure_moment=pressure_force.copy()
        for i,(pad,frame,arm) in enumerate(contact_geometry):
            action=np.zeros((3,size));action[0,:nv]=-Dn[i]
            action[1:,nv+nu+2*i:nv+nu+2*i+2]=np.eye(2)
            action=frame.T@action;bias=frame[0]*p0[i]
            force[pad]+=action;constant[pad]+=bias
            pressure_force[pad,:,i]=frame[0]
            cross=np.array([[0.,-arm[2],arm[1]],[arm[2],0.,-arm[0]],[-arm[1],arm[0],0.]])
            moment[pad]+=cross@action;moment_constant[pad]+=cross@bias
            pressure_moment[pad,:,i]=cross@frame[0]
        scaling=np.r_[self.speed,self.limit,np.full(nt+ns,1.)]
        A=np.vstack(rows)*scaling;lo=np.concatenate(low);hi=np.concatenate(high)
        equal=np.isfinite(lo)&(lo==hi);upper=(~equal)&np.isfinite(hi);lower=(~equal)&np.isfinite(lo)
        matrix=np.vstack([A[equal],A[upper],-A[lower]]);bound=np.r_[hi[equal],hi[upper],-lo[lower]]
        cones=[clarabel.ZeroConeT(int(sum(equal))),clarabel.NonnegativeConeT(int(sum(upper)+sum(lower)))]
        for maps,biases,limit in [(force,constant,30.),(moment,moment_constant,5.)]:
            for mapping,bias in zip(maps,biases):
                matrix=np.vstack([matrix,np.zeros(size),-mapping*scaling]);bound=np.r_[bound,limit,bias]
                cones.append(clarabel.SecondOrderConeT(4))
        settings=clarabel.DefaultSettings();settings.verbose=False
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
        solver_note=None;pressure_correction=np.zeros(ne)
        if self.fixed_input is not None:
            # A fixed actual command determines the one-period velocity and
            # contact response. Solve those equations, then check every
            # original bound/cone; this is not another input allocation.
            from build.held_input_solver import violation
            command=np.asarray(self.fixed_input)
            initial_effort=command.copy()
            initial_effort[:nj]=np.where(self.dock,command[:nj],self.kp*command[:nj])-self.kp*q[self.qi]-self.kd*v[self.vi]
            z=np.zeros(size);z[nv:nv+nu]=initial_effort/self.limit
            unknown=np.r_[np.arange(nv),np.arange(nv+nu,size)]
            count=cones[0].dim
            if count!=len(unknown):raise ValueError('fixed-command response requires independent full dynamics')
            z[unknown]=np.linalg.solve(matrix[:count,unknown],bound[:count]-matrix[:count]@z)
            error=violation(matrix,bound,cones,z)
            if not np.isfinite(error) or error>1e-5:raise ValueError('fixed-command contact conditions: '+str(error))
            iterations=0;solver_note=dict(original_residual=error,fixed_command=True)
        elif self.project_input:
            if input_reference is None or not(self.regularized_friction and self.finite_support):
                raise ValueError('input projection requires a reference and finite contact response')
            from build.projected_contact_input import project
            options={}
            if self.project_object:
                ov=int(m.jnt_dofadr[m.joint('object').id]);indices=np.arange(ov,ov+6)
                options=dict(motion_indices=indices,motion_goal=goal[indices]/self.speed[indices],input_weight=self.input_weight)
            z,solver_note=project(matrix,bound,cones,nv,nu,target[nv:nv+nu]/scaling[nv:nv+nu],**options)
            iterations=solver_note['iterations']
        elif self.support_force_priority and ns:
            if not self.finite_support:raise ValueError('support priority requires finite support response')
            from build.support_priority_solver import solve
            z,solver_note=solve(objective*scaling[:,None]*scaling[None,:],linear*scaling,matrix,bound,cones,nv,nu,nt,ns)
            iterations=solver_note['iterations']
        elif self.condensed_solve:
            from build.contact_condensed_solver import solve
            try:
                z,solver_note=solve(objective*scaling[:,None]*scaling[None,:],linear*scaling,matrix,bound,cones,nv,nu,nt,
                    fixed_zero_tail=ns if self.finite_support and self.support_load_upper==0. else 0)
            except ValueError:
                if not(ne and self.regularized_friction) or self.friction_reserve:raise
                from build.contact_friction_modes import solve as solve_modes
                pressure_rows=np.vstack([pressure_blocks.get(i,np.zeros((len(row),ne)))for i,row in enumerate(rows)])
                pressure_matrix=np.vstack([pressure_rows[equal],pressure_rows[upper],-pressure_rows[lower]])
                for maps in [pressure_force,pressure_moment]:
                    for mapping in maps:pressure_matrix=np.vstack([pressure_matrix,np.zeros(ne),-mapping])
                z,pressure_correction,solver_note=solve_modes(objective*scaling[:,None]*scaling[None,:],linear*scaling,
                    matrix,bound,cones,nv,nu,nt,pressure_matrix,Dn*scaling[:nv],p0,rn,m.opt.impratio,v/scaling[:nv])
            iterations=solver_note['iterations']
        elif self.fast_solve:
            from build.held_input_solver import solve
            z,solver_note=solve(objective*scaling[:,None]*scaling[None,:],linear*scaling,matrix,bound,cones)
            iterations=None
        else:
            solver=clarabel.DefaultSolver(sp.csc_matrix(np.triu(objective*scaling[:,None]*scaling[None,:])),linear*scaling,
                sp.csc_matrix(matrix),bound,cones,settings)
            solution=solver.solve()
            if solution.status not in [clarabel.SolverStatus.Solved,clarabel.SolverStatus.AlmostSolved]:
                raise ValueError('contact step '+str(solution.status))
            z=np.array(solution.x);iterations=int(solution.iterations)
        x=z*scaling;pred=x[:nv];effort=x[nv:nv+nu]
        evaluated=A@z;check_rows=np.ones(len(lo),dtype=bool)
        if solver_note and 'friction_modes' in solver_note:
            evaluated+=pressure_rows@pressure_correction
            # The boundary solve independently rechecked the replacement
            # constitutive equations and weighted force projection.
            equal_ids=np.flatnonzero(equal);tangent_ids=equal_ids[nv:nv+nt]
            check_rows[tangent_ids]=False
        residual=max(float(np.max((lo-evaluated)[check_rows])),float(np.max((evaluated-hi)[check_rows])))
        if residual>1e-5:raise ValueError('contact step residual '+str(residual))
        predicted_force=np.einsum('ijk,k->ij',force,x)+constant+pressure_force@pressure_correction
        predicted_moment=np.einsum('ijk,k->ij',moment,x)+moment_constant+pressure_moment@pressure_correction
        if np.max(np.linalg.norm(predicted_force,axis=1))>30.00001 or np.max(np.linalg.norm(predicted_moment,axis=1))>5.00001:
            raise ValueError('contact step pad load residual')
        loss=.5*float(np.sum(weights*(x-target)**2))
        if normal_targets is not None:loss+=.5*float(np.sum(((jacobian@pred-rate)/scale)**2))
        cmd=(effort[:nj]+self.kp*q[self.qi]+self.kd*v[self.vi])/np.where(self.dock,1.,self.kp)
        self.diagnostics=dict(gripper_elements=ne,support_elements=len(mus),free_object_velocity_observation=free_object,iterations=iterations,solver_note=solver_note,
                              predicted_support_normal_load=float(x[nv+nu+nt:nv+nu+nt+ns:3].sum()),
                              constraint_error=residual,objective_loss=loss,observed_tangent_speed=float(np.max(np.linalg.norm((Ct@v).reshape(-1,2),axis=1),initial=0.)),normal_force=(p0-Dn@pred+pressure_correction).tolist(),
                              pad_force=predicted_force.tolist(),pad_moment=predicted_moment.tolist(),
                              pad_gaps=None if pad_gaps is None else pad_gaps.tolist(),
                              velocity_error=float(np.linalg.norm((pred-goal)/self.speed)),
                              target_velocity=goal.tolist(),predicted_velocity=pred.tolist())
        return np.r_[cmd,effort[nj:]]
