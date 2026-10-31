"""Fixed-pose load screen; its solution is NOT a realized contact/control plan."""
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize, LinearConstraint, Bounds
from .linear_program import linprog
from scipy.linalg import qr


def point_jacobian(model,data,q,frame,point):
    J=pin.computeFrameJacobian(model,data,q,frame,pin.LOCAL_WORLD_ALIGNED)
    arm=data.oMf[frame].rotation@np.asarray(point)
    return J[:3]-pin.skew(arm)@J[3:]


def contact_basis(contact):
    n=np.asarray(contact['normal'],dtype=float)
    if abs(float(n@n)**.5-1)>1e-9:raise ValueError('non-unit normal')
    if 'tangent' in contact:
        t=np.asarray(contact['tangent'],dtype=float)
        if abs(n@t)>1e-9 or abs(float(t@t)**.5-1)>1e-9:raise ValueError('invalid material tangent')
    else:
        # This is a single three-vector, not a broadcasted cross product.
        k=int(np.argmin(abs(n)))
        t=np.array([(0.,n[2],-n[1]),(-n[2],0.,n[0]),(n[1],-n[0],0.)][k]);t/=float(t@t)**.5
    b=np.array([n[1]*t[2]-n[2]*t[1],n[2]*t[0]-n[0]*t[2],n[0]*t[1]-n[1]*t[0]])
    return np.column_stack([n,t,b])


def contact_normal_obstruction(contacts, required_force, minimum_pad_normal_force_n):
    """A separating force projection for the SAME inner contact model.

    If every force cone projects positively on an axis, pad normal-load floors
    impose a positive lower bound on the sum. A smaller required projection
    cannot be repaired by changing rotor directions. Passing this cheap check
    says nothing about full wrench or robot feasibility. Curved/opposed pads
    remain eligible whenever no separating axis is found.
    """
    if not contacts or minimum_pad_normal_force_n<=0:return None
    normals=np.asarray([c['normal'] for c in contacts]);mu=np.array([c['mu'] for c in contacts])
    axes=np.eye(3)[np.argmin(abs(normals),axis=1)]
    t1=np.cross(normals,axes);t1/=np.linalg.norm(t1,axis=1)[:,None]
    for i,c in enumerate(contacts):
        if 'tangent' in c:t1[i]=contact_basis(c)[:,1]
    t2=np.cross(normals,t1);pads={c['frame'] for c in contacts}
    for axis in normals:
        # Minimum projection per unit normal force over the diamond cone.
        coefficients=normals@axis-mu*np.maximum(abs(t1@axis),abs(t2@axis))
        if coefficients.min()<=0:continue
        lower=minimum_pad_normal_force_n*sum(min(coefficients[i] for i,c in enumerate(contacts) if c['frame']==pad) for pad in pads)
        required=float(np.asarray(required_force)@axis)
        if required<lower-1e-8*(1+abs(required)+lower):
            return dict(scope='current_material_inner_force_model_only',axis=axis.tolist(),
                        required_projection_n=required,minimum_projection_n=float(lower))
    return None


def object_wrench_obstruction(contacts,required_force,rotation,minimum_normal=.5):
    """A dual lower bound for an object-only relaxation of the trim screen.

    Rotor directions cannot repair unbalanced object force/torque. Dropping
    robot equilibrium and relaxing pad L1 caps to normal caps only enlarges
    the original feasible set. A positive certified residual rejects this
    fixed contact geometry, never the graph or other contact positions.
    """
    if not contacts or any(not c.get('target_frame') for c in contacts):return None
    # This certificate minimizes over finite force boxes. An unbounded
    # contact needs a different certificate, not an invented finite cap.
    if any(c['max_force'] is None or not np.isfinite(c['max_force']) for c in contacts):return None
    columns=[];pads={};upper=[]
    for i,c in enumerate(contacts):
        n,t,b=contact_basis(c).T
        forces=np.array([n+c['mu']*t,n-c['mu']*t,n+c['mu']*b,n-c['mu']*b]).T
        arm=np.asarray(rotation)@np.asarray(c['target_point'])
        columns.append(np.vstack([forces,np.cross(arm,forces.T).T]))
        upper.extend([float(c['max_force'])]*4)
        pads.setdefault(c['frame'],[]).append(i)
    W=np.concatenate(columns,axis=1);target=np.r_[required_force,np.zeros(3)]
    # Scale only the residual objective; force constraints keep their units.
    scale=np.maximum(np.linalg.norm(W,axis=1),1.)
    width=W.shape[1]+1
    A=[np.c_[W/scale[:,None],-np.ones(6)],np.c_[-W/scale[:,None],-np.ones(6)]]
    rhs=[target/scale,-target/scale]
    for i,c in enumerate(contacts):
        row=np.zeros(width);row[4*i:4*i+4]=1.
        A.append(row[None]);rhs.append([c['max_force']])
    for ids in pads.values():
        row=np.zeros(width)
        for i in ids:row[4*i:4*i+4]=1.
        A.append(-row[None]);rhs.append([-minimum_normal])
        limits=[contacts[i]['pad_force_limit'] for i in ids if 'pad_force_limit' in contacts[i]]
        if limits:A.append(row[None]);rhs.append([min(limits)])
    A=np.vstack(A);rhs=np.concatenate(rhs);cost=np.zeros(width);cost[-1]=1.
    result=linprog(cost,A_ub=A,b_ub=rhs,bounds=[(0,u) for u in upper]+[(0,None)])
    if not result.success or result.fun<1e-7:return None
    dual=np.maximum(-result.ineqlin.marginals,0.)
    # Feasible dual weights plus exact minimization of the Lagrangian over
    # finite force-variable boxes certify a lower bound even after rounding.
    dual/=max(float(dual[:12].sum()),1.)*(1.+1e-12)
    gradient=cost+A.T@dual
    if gradient[-1]<0:return None
    bound=float(-dual@rhs+np.minimum(gradient[:-1],0.)@np.asarray(upper))
    if not np.isfinite(bound) or bound<=1e-6:return None
    return dict(scope='fixed_material_object_wrench_relaxation',residual_lower_bound=bound)


def _minimum_burden(E,equality,A,b,bounds,nr,peak_bound,reusable=None,strict=False,power=1.5):
    """Convex force proposal; the quadratic variant is not a power optimum."""
    import clarabel
    import scipy.sparse as sp
    if power not in [1.5,2.]:raise ValueError('unsupported force objective')
    nx=len(bounds)+1;total=nx+(nr if power==1.5 else 0)
    lower=np.array([-np.inf if lo is None else lo for lo,_ in bounds]+[0.])
    upper=np.array([np.inf if hi is None else hi for _,hi in bounds]+[peak_bound])
    cache_key=(peak_bound,power)
    cached=None if reusable is None else reusable.get(cache_key)
    if cached is not None:
        solver,rhs=cached;rhs[:len(E)]=equality
        solver.update(b=rhs);solved=solver.solve()
    else:
        unit=np.eye(nx);upper_ids=np.isfinite(upper);lower_ids=np.isfinite(lower)
        linear=np.vstack([E,np.asarray(A),unit[upper_ids],-unit[lower_ids]])
        rhs=np.r_[equality,b,upper[upper_ids],-lower[lower_ids]]
        cones=[clarabel.ZeroConeT(len(E)),clarabel.NonnegativeConeT(len(linear)-len(E))]
        if power==1.5:
            con=np.zeros((len(linear)+3*nr,total));con[:len(linear),:nx]=linear
            rhs=np.r_[rhs,np.tile([0.,1.,0.],nr)]
            for i in range(nr):
                con[len(linear)+3*i,nx+i]=-1.;con[len(linear)+3*i+2,i]=-1.
            objective=np.r_[np.zeros(nx),np.ones(nr)];H=sp.csc_matrix((total,total))
            cones.extend(clarabel.PowerConeT(2/3) for _ in range(nr))
        else:
            con=linear;objective=np.zeros(total)
            H=sp.diags(np.r_[np.full(nr,2.),np.zeros(total-nr)],format='csc')
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_feas=1e-10 if strict else 1e-9
        settings.tol_gap_abs=settings.tol_gap_rel=1e-10 if strict else 1e-8
        solver=clarabel.DefaultSolver(H,objective,sp.csc_matrix(con),rhs,cones,settings)
        if reusable is not None:reusable[cache_key]=(solver,rhs.copy())
        solved=solver.solve()
    # Remove solver roundoff at bounds, then verify ALL force equations on
    # the command we actually return (especially nonnegative rotor thrust).
    x=np.clip(np.asarray(solved.x)[:nx],lower,upper)
    error=max(float(abs(E@x-equality).max(initial=0.)),
        float((np.asarray(A)@x-np.asarray(b)).max(initial=0.)),
        float((lower-x).max(initial=0.)),float((x-upper).max(initial=0.)))
    from types import SimpleNamespace
    if not strict and str(solved.status) in ['Solved','AlmostSolved'] and error>=1e-7:
        return _minimum_burden(E,equality,A,b,bounds,nr,peak_bound,strict=True,power=power)
    success=str(solved.status) in ['Solved','AlmostSolved'] and error<1e-7
    return SimpleNamespace(x=x,success=success,status=0 if success else 1,message=str(solved.status),nit=solved.iterations,dual=np.asarray(solved.z))


def static_balance(model,meta,q,contacts=(),time_limit_s=None, *, payload_fraction=1., balance_thrust=False,object_wrench=None,base_load=None,base_acceleration=None,peak_slack=None,minimum_pad_normal_force_n=0.,gimbal_gradient=False,object_support=None,rotor_rays=None,rotor_direction_cost=None,minimize_burden=False,fast_lp=False,direct_burden=False,workspace=None,direct_gradient=False,direction_model=False,burden_power=1.5,relaxed_actuator_scale=False,rotor_polytope=None):
    """Contacts give force on the robot; the opposite force acts on the object.

    normal, point and target_point are expressed in world/each local frame as
    indicated. A diamond inner friction cone and l1 force cap are conservative.
    Failure of this screen does not prove physical infeasibility.
    """
    solve_lp=linprog
    if relaxed_actuator_scale and rotor_rays is None and rotor_polytope is None:raise ValueError('actuator scaling is only a direction proposal for growth')
    if rotor_polytope is not None and (not relaxed_actuator_scale or rotor_rays is not None or workspace is not None or direct_burden or gimbal_gradient):
        raise ValueError('compact rotor polytope is only a growth capacity proposal')
    if fast_lp:
        if gimbal_gradient:raise ValueError('fast LP does not provide sensitivity gradients')
        from .linear_program import conic_linprog
        solve_lp=conic_linprog
    if peak_slack is not None and (not np.isfinite(peak_slack) or peak_slack<0):raise ValueError('invalid peak-effort slack')
    if not np.isfinite(minimum_pad_normal_force_n) or minimum_pad_normal_force_n<0:raise ValueError('invalid minimum pad load')
    start=time.perf_counter();data=model.createData();pin.framesForwardKinematics(model,data,q)
    gravity=pin.computeGeneralizedGravity(model,data,q).copy()
    if base_acceleration is not None:
        acceleration=np.asarray(base_acceleration,dtype=float)
        if len(meta['bases'])!=1 or acceleration.shape!=(6,) or not np.isfinite(acceleration).all():
            raise ValueError('base acceleration requires one connected base and six finite components')
        av=np.zeros(model.nv);bv=meta['bases'][0]['v'];av[bv:bv+6]=acceleration
        gravity=pin.rnea(model,data,q,np.zeros(model.nv),av).copy()
    if base_load is not None:
        load=np.asarray(base_load,dtype=float)
        if len(meta['bases'])!=1 or load.shape!=(6,) or not np.isfinite(load).all():
            raise ValueError('base load requires one connected base and six finite generalized components')
        bv=meta['bases'][0]['v'];gravity[bv:bv+6]=load
    if meta['object'] is not None:
        ov=meta['object']['v'];gravity[ov:ov+6]*=payload_fraction
        if object_wrench is not None:
            wrench=np.asarray(object_wrench,dtype=float)
            if wrench.shape!=(6,) or not np.isfinite(wrench).all():raise ValueError('invalid object wrench')
            R=data.oMf[meta['object']['frame']].rotation
            gravity[ov:ov+6]+=np.r_[R.T@wrench[:3],R.T@wrench[3:]]
    # Only equality loads may vary when reusing this exact physical pose.
    # The key includes every geometric/inequality input and actuator bias.
    workspace_key=None if workspace is None else (id(model),id(meta),np.asarray(q).tobytes(),
        tuple((c['frame'],c.get('target_frame'),tuple(c['normal']),tuple(c.get('tangent',())),tuple(c['point']),tuple(c.get('target_point',())),c['mu'],c['max_force'],c.get('pad_force_limit')) for c in contacts),
        None if object_support is None else np.asarray(object_support).tobytes(),
        None if rotor_rays is None else np.asarray(rotor_rays).tobytes(),
        minimum_pad_normal_force_n,gravity[np.array([a['v'] for a in meta['actuators']])].tobytes())
    if workspace is not None and rotor_rays is not None:raise ValueError('workspace needs actual rotor directions')
    prepared=None if workspace is None else workspace.get(workspace_key)
    if prepared is None:
        columns=[];bounds=[];contact_columns=[];frame_jacobians={}
        pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
        def frame_jacobian(frame):
            if frame not in frame_jacobians:frame_jacobians[frame]=pin.getFrameJacobian(model,data,frame,pin.LOCAL_WORLD_ALIGNED)
            return frame_jacobians[frame]
        def material_jacobian(frame,point):
            jac=frame_jacobian(frame);arm=data.oMf[frame].rotation@np.asarray(point)
            return jac[:3]-pin.skew(arm)@jac[3:]
        for rotor in meta['rotors']:
            frame=rotor['frame'];J=frame_jacobian(frame)
            columns.append(J[:3].T@data.oMf[frame].rotation[:,2]);bounds.append((rotor['minimum'],rotor['maximum']))
        if rotor_rays is not None:
            rays=np.asarray(rotor_rays);nr=len(meta['rotors'])
            if rays.ndim!=3 or rays.shape[1:]!=(model.nv,nr) or balance_thrust or gimbal_gradient:
                raise ValueError('rotor direction rays are for a static proposal only')
            columns=[rays[k,:,i] for i in range(nr) for k in range(len(rays))]
            bounds=[(0.,None if relaxed_actuator_scale else r['maximum']) for r in meta['rotors'] for _ in rays]
        if rotor_polytope is not None:
            basis,facets=map(np.asarray,rotor_polytope);nr=len(meta['rotors'])
            if basis.shape!=(2,model.nv,nr) or facets.ndim!=2 or facets.shape[1]!=3:
                raise ValueError('invalid compact rotor polytope')
            columns=[basis[k,:,i] for i in range(nr) for k in range(2)]
            bounds=[(None,None)]*(2*nr)
        for c in contacts:
            R=contact_basis(c);J=material_jacobian(meta['frames'][c['frame']],c['point'])
            if c.get('target_frame'):
                J-=material_jacobian(meta['frames'][c['target_frame']],c['target_point'])
            first=len(columns);contact_columns.append((first,c,R))
            columns.extend((J.T@R)[:,i] for i in range(3))
            bounds.extend([(0,c['max_force']), (None,None),(None,None)])
        if object_support is not None:
            support=np.asarray(object_support,dtype=float)
            if meta['object'] is None or support.ndim!=2 or support.shape[0]!=6 or not np.isfinite(support).all():
                raise ValueError('object support must contain world wrench directions')
            rotation=data.oMf[meta['object']['frame']].rotation
            for direction in support.T:
                column=np.zeros(model.nv)
                column[ov:ov+6]=np.r_[rotation.T@direction[:3],rotation.T@direction[3:]]
                columns.append(column);bounds.append((None,None))
        # A finite pad has one force budget. Optional aggregate budgets avoid
        # imposing an equal share on every mesh/contact sample. The auxiliary
        # variables bound absolute tangential forces in the same conservative
        # L1 norm already used by the per-point screen.
        pad_limits={};absolute_columns={}
        for first,c,_ in contact_columns:
            if 'pad_force_limit' not in c:continue
            limit=float(c['pad_force_limit']);name=c['frame']
            if not np.isfinite(limit) or limit<=0 or (name in pad_limits and pad_limits[name]!=limit):
                raise ValueError('inconsistent aggregate pad force limit')
            pad_limits[name]=limit
        for first,c,_ in contact_columns:
            if c['frame'] in pad_limits:
                absolute_columns[first]=len(columns)
                columns.extend([np.zeros(model.nv),np.zeros(model.nv)])
                bounds.extend([(0,None),(0,None)])
        B=np.column_stack(columns);nv=model.nv;nx=B.shape[1]+1
        actuated=np.array([a['v'] for a in meta['actuators']]);unactuated=np.setdiff1d(np.arange(nv),actuated)
        E=np.zeros((len(unactuated),nx));E[:,:-1]=B[unactuated]
        A=[];b=[]
        if rotor_rays is not None:
            for i,r in enumerate(meta['rotors']):
                row=np.zeros(nx);row[i*len(rays):(i+1)*len(rays)]=1.
                if relaxed_actuator_scale:row[-1]=-r['maximum']
                A.append(row);b.append(0. if relaxed_actuator_scale else r['maximum'])
        if rotor_polytope is not None:
            for i,r in enumerate(meta['rotors']):
                for facet in facets:
                    row=np.zeros(nx);row[2*i:2*i+2]=facet[:2];row[-1]=facet[2]*r['maximum']
                    A.append(row);b.append(0.)
        for a in meta['actuators']:
            row=np.zeros(nx);row[:-1]=-B[a['v']];row[-1]=-a['effort']
            A.append(row);b.append(-gravity[a['v']])
            row=row.copy();row[:-1]*=-1;A.append(row);b.append(gravity[a['v']])
        for first,c,R in contact_columns:
            for sx in [-1,1]:
                for sy in [-1,1]:
                    row=np.zeros(nx);row[first:first+3]=[-c['mu'],sx,sy];A.append(row);b.append(0.)
                    if c['max_force'] is not None and np.isfinite(c['max_force']):
                        row=row.copy();row[first]=1.;A.append(row);b.append(c['max_force'])
            if first in absolute_columns:
                aux=absolute_columns[first]
                for tangent in range(2):
                    for sign in (-1,1):
                        row=np.zeros(nx);row[first+1+tangent]=sign;row[aux+tangent]=-1.
                        A.append(row);b.append(0.)
        # A pad may have many material points. Require its SUM, not a minimum
        # on every point or merely the total across all selected pads.
        pad_columns={}
        for first,c,_ in contact_columns:pad_columns.setdefault(c['frame'],[]).append(first)
        for name,limit in pad_limits.items():
            row=np.zeros(nx)
            for first in pad_columns[name]:
                row[first]=1.;row[absolute_columns[first]:absolute_columns[first]+2]=1.
            A.append(row);b.append(limit)
        if minimum_pad_normal_force_n:
            for indices in pad_columns.values():
                row=np.zeros(nx);row[indices]=-1.
                A.append(row);b.append(-minimum_pad_normal_force_n)
        prepared=(B,nv,nx,bounds,contact_columns,actuated,unactuated,E,np.asarray(A),np.asarray(b),pad_columns,{})
        if workspace is not None:workspace[workspace_key]=prepared
    B,nv,nx,bounds,contact_columns,actuated,unactuated,E,A,b,pad_columns,solvers=prepared
    objective=np.zeros(nx);objective[-1]=1.
    if rotor_polytope is None:objective[:len(meta['rotors'])*(len(rays) if rotor_rays is not None else 1)]=1e-8
    options={'maxiter':500}
    if time_limit_s is not None:options['time_limit']=float(time_limit_s)
    if direct_burden:
        if rotor_rays is not None or gimbal_gradient or balance_thrust:
            raise ValueError('direct burden needs actual rotor directions and no secondary objective')
        solution=_minimum_burden(E,gravity[unactuated],A,b,bounds,len(meta['rotors']),1.,reusable=solvers if workspace is not None else None,power=burden_power)
    else:
        solution=solve_lp(objective,A_ub=np.array(A),b_ub=np.array(b),A_eq=E,b_eq=gravity[unactuated],bounds=bounds+[(0,None)],
                         method='highs',options=options)
    result=dict(scope='fixed_pose_inner_force_screen_only',task_accepted=False,
                solver_status=int(solution.status),solver_message=solution.message,iterations=int(solution.nit),
                wall_s=time.perf_counter()-start,physical_impossibility_proven=False)
    if direct_burden:result['force_objective_power']=burden_power
    if not solution.success:
        result['status']='unresolved_or_inner_model_infeasible';return result
    x=solution.x
    result['feasible_peak_effort_ratio' if direct_burden else 'minimum_peak_effort_ratio']=float(x[-1])
    if rotor_polytope is not None:
        result.update(status='direction_proposal',scope='relaxed_motor_scale_for_growth_not_feasibility')
        return result
    if rotor_rays is not None:
        if relaxed_actuator_scale:result['scope']='relaxed_motor_scale_for_growth_not_feasibility'
        if rotor_direction_cost is not None and x[-1]<=1.:
            secondary=np.zeros(nx);secondary[:nr*len(rays)]=np.asarray(rotor_direction_cost).ravel();secondary[-1]=.001
            smoother=solve_lp(secondary,A_ub=np.array(A),b_ub=np.array(b),A_eq=E,b_eq=gravity[unactuated],
                bounds=bounds+[(0,max(.95,float(x[-1])+1e-7))],method='highs',options=options)
            if smoother.success:x=smoother.x
        result.update(status='direction_proposal',rotor_coefficients=x[:nr*len(rays)].reshape(nr,-1).tolist())
        return result
    if direct_gradient or direction_model:
        if not direct_burden:raise ValueError('burden sensitivity requires the direct convex objective')
        dual=np.zeros(nv);dual[unactuated]=-solution.dual[:len(E)]
        z=solution.dual[len(E):len(E)+len(A)]
        for i,a in enumerate(meta['actuators']):dual[a['v']]=z[2*i]-z[2*i+1]
        from .rotor_distribution import rotor_actions
        gimbals=[meta['actuators'][r['gimbal']] for r in meta['rotors']]
        qi=np.array([a['q'] for a in gimbals]);vi=np.array([a['v'] for a in gimbals])
        cached=solvers.get('direction_derivatives')
        if cached is None:
            gravity_derivative=pin.computeGeneralizedGravityDerivatives(model,data,q)[:,vi].copy()
            plus=q.copy();minus=q.copy();plus[qi]+=1e-6;minus[qi]-=1e-6
            force_derivative=(rotor_actions(model,meta,plus)-rotor_actions(model,meta,minus))/2e-6
            cached=gravity_derivative,force_derivative
            solvers['direction_derivatives']=cached
        changes=cached[0]-cached[1]*x[:len(qi)]
        result['burden_direction_gradient']=(dual@changes).tolist()
        if direction_model:
            result['direction_model']=dict(E=E,equality=gravity[unactuated],A=np.asarray(A),b=np.asarray(b),
                bounds=bounds,change=changes,unactuated=unactuated,actuated=actuated,x=x.copy())

    if gimbal_gradient:
        # Envelope derivative of the solved force LP. Contact geometry is
        # independent of the rotor-only gimbal joints. Original constraints
        # and a fresh LP are still used to accept every changed posture.
        dual=np.zeros(nv);dual[unactuated]=solution.eqlin.marginals
        marginals=solution.ineqlin.marginals
        for i,a in enumerate(meta['actuators']):dual[a['v']]=-marginals[2*i]+marginals[2*i+1]
        gravity_derivative=pin.computeGeneralizedGravityDerivatives(model,data,q).copy()
        gradient=[]
        for i,rotor in enumerate(meta['rotors']):
            a=meta['actuators'][rotor['gimbal']];frame=rotor['frame'];columns=[]
            for sign in [1.,-1.]:
                moved=q.copy();moved[a['q']]+=sign*1e-6
                pin.framesForwardKinematics(model,data,moved)
                J=pin.computeFrameJacobian(model,data,moved,frame,pin.LOCAL_WORLD_ALIGNED)
                columns.append(J[:3].T@data.oMf[frame].rotation[:,2])
            gradient.append(float(dual@(gravity_derivative[:,a['v']]-x[i]*(columns[0]-columns[1])/2e-6)))
        result['gimbal_peak_gradient']=gradient

    if minimize_burden and not direct_burden and x[-1]<=1.+1e-7:
        cap=min(1.,max(.8,float(x[-1])+.02))
        refined=_minimum_burden(E,gravity[unactuated],A,b,bounds,len(meta['rotors']),cap)
        nr=len(meta['rotors'])
        accepted=refined.success and np.maximum(refined.x[:nr],0.).dot(np.sqrt(np.maximum(refined.x[:nr],0.)))<=np.maximum(x[:nr],0.).dot(np.sqrt(np.maximum(x[:nr],0.)))+1e-7
        if accepted:x=refined.x
        result['burden_refinement_success']=accepted
        result['secondary_peak_bound']=cap
    elif balance_thrust:
        # Keep the verified LP constraints, then choose a less concentrated
        # rotor trim. This plans a reference; it does not set contact forces.
        nr=len(meta['rotors']);limits=np.array([a['effort'] for a in meta['actuators']])
        Qj=B[actuated]/limits[:,None];gj=gravity[actuated]/limits
        f0=(meta['mass']-(meta['object'] or {}).get('mass',0)*(1-payload_fraction))*9.80665/nr
        def cost(z):
            residual=Qj@z[:-1]-gj
            return 10*residual@residual+.02*np.sum((z[:nr]-f0)**2)+.001*np.sum(z[nr:-1]**2)
        def gradient(z):
            g=np.zeros(nx);g[:-1]=20*Qj.T@(Qj@z[:-1]-gj)
            g[:nr]+=.04*(z[:nr]-f0);g[nr:-1]+=.002*z[nr:-1];return g
        _,rr,pivot=qr(E.T,pivoting=True,mode='economic')
        independent=pivot[:np.linalg.matrix_rank(rr,tol=1e-10)]
        lower=np.array([-np.inf if a is None else a for a,_ in bounds]+ [0.])
        peak_bound=max(.8,float(x[-1])+.02) if peak_slack is None else min(1.,float(x[-1])+peak_slack)
        result['secondary_peak_bound']=peak_bound
        upper=np.array([np.inf if b is None else b for a,b in bounds]+[peak_bound])
        import clarabel
        import scipy.sparse as sp
        from types import SimpleNamespace
        H=np.zeros((nx,nx));H[:-1,:-1]=20*Qj.T@Qj
        H[np.arange(nr),np.arange(nr)]+=.04
        ids=np.arange(nr,nx-1);H[ids,ids]+=.002
        g=np.zeros(nx);g[:-1]=-20*Qj.T@gj;g[:nr]-=.04*f0
        unit=np.eye(nx);upper_ids=np.isfinite(upper);lower_ids=np.isfinite(lower)
        eq=E[independent];rhs=gravity[unactuated][independent]
        con=np.vstack([eq,np.asarray(A),unit[upper_ids],-unit[lower_ids]])
        rhs=np.r_[rhs,b,upper[upper_ids],-lower[lower_ids]]
        scale=np.maximum(np.linalg.norm(con,axis=1),1.)
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-10
        solved=clarabel.DefaultSolver(sp.csc_matrix(np.triu(H)),g,sp.csc_matrix(con/scale[:,None]),rhs/scale,
            [clarabel.ZeroConeT(len(eq)),clarabel.NonnegativeConeT(len(con)-len(eq))],settings).solve()
        candidate=np.asarray(solved.x)
        residual=max(float(abs(E@candidate-gravity[unactuated]).max()),float((np.asarray(A)@candidate-np.asarray(b)).max()),
                     float((lower-candidate).max()),float((candidate-upper).max()))
        accepted=str(solved.status) in ['Solved','AlmostSolved'] and residual<1e-7 and cost(candidate)<=cost(x)+1e-8
        refined=SimpleNamespace(success=accepted,x=candidate)
        if not accepted:
            refined=minimize(cost,x,jac=gradient,method='SLSQP',bounds=Bounds(lower,upper),
                constraints=[LinearConstraint(E[independent],gravity[unactuated][independent],gravity[unactuated][independent]),
                             LinearConstraint(np.array(A),-np.inf,np.array(b))],
                options=dict(maxiter=150,ftol=1e-9))
        if refined.success and refined.x[-1]<=peak_bound+1e-8:x=refined.x
        result['balanced_refinement_success']=bool(refined.success)
    # Remove sub-tolerance bound noise before returning a physical command.
    # Recompute balance with the corrected vector; never accept a materially
    # negative thrust or keep a certificate for the uncorrected command.
    nr=len(meta['rotors']);low=np.array([r['minimum'] for r in meta['rotors']]);high=np.array([r['maximum'] for r in meta['rotors']])
    rotor_violation=max(float(np.max(low-x[:nr],initial=0.)),float(np.max(x[:nr]-high,initial=0.)))
    if rotor_violation<=1e-7:x[:nr]=np.clip(x[:nr],low,high)
    torque=gravity-B@x[:-1]
    effort_ratio=max(abs(float(torque[a['v']]))/a['effort'] for a in meta['actuators'])
    residual=float(np.max(abs(torque[unactuated]),initial=0.))
    violation=max(float(np.max(A@x-b,initial=0.)),rotor_violation)
    result['wall_s']=time.perf_counter()-start
    result.update(status='static_screen_pass' if x[-1]<=1.+1e-7 and residual<1e-7 and violation<1e-7 else 'static_screen_violation',
                  max_normalized_joint_effort=effort_ratio,equilibrium_residual=residual,inequality_residual=violation,
                  thrust=x[:len(meta['rotors'])].tolist(),
                  minimum_pad_normal_force_n=float(minimum_pad_normal_force_n),
                  pad_normal_forces_n={name:float(x[indices].sum()) for name,indices in pad_columns.items()},
                  joints=[dict(name=a['name'],effort_nm=float(torque[a['v']]),limit_nm=a['effort'],normalized=abs(float(torque[a['v']]))/a['effort']) for a in meta['actuators']],
                  contacts=[dict(frame=c['frame'],force_world=(R@x[first:first+3]).tolist(),normal_force=float(x[first])) for first,c,R in contact_columns])
    return result


def free_support_distribution(model,meta,q,thrust):
    """Preserve a known commanded base wrench while screening free-arm loads.

    This frozen-pose allocation does not establish a stationary free robot or
    measure the physical contact loads during the transition.
    """
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    force=np.asarray(thrust,dtype=float)
    if force.shape!=(len(meta['rotors']),) or not np.isfinite(force).all():raise ValueError('invalid rotor command')
    load=np.zeros(model.nv)
    for rotor,value in zip(meta['rotors'],force):
        frame=rotor['frame'];J=pin.computeFrameJacobian(model,data,q,frame,pin.LOCAL_WORLD_ALIGNED)
        load+=J[:3].T@data.oMf[frame].rotation[:,2]*value
    bv=meta['bases'][0]['v']
    # Once contact vanishes the surplus base wrench accelerates the robot.
    # Include that predicted inertial joint load; do not call the loaded base
    # wrench a free-hover equilibrium. Joint accelerations are zero in this
    # local endpoint prediction, not asserted to be zero in actual execution.
    gravity=pin.computeGeneralizedGravity(model,data,q).copy();mass=pin.crba(model,data,q).copy()
    mass=np.triu(mass)+np.triu(mass,1).T
    acceleration=np.linalg.solve(mass[bv:bv+6,bv:bv+6],load[bv:bv+6]-gravity[bv:bv+6])
    result=static_balance(model,meta,q,payload_fraction=0.,balance_thrust=True,base_load=load[bv:bv+6],base_acceleration=acceleration)
    result['scope']='frozen_pose_free_base_acceleration_joint_load_screen_same_commanded_base_wrench'
    result['base_load']=load[bv:bv+6].tolist()
    result['free_base_acceleration']=acceleration.tolist()
    return result


def restoring_screen(model,meta,q,contacts,angular_acceleration=.3, *, minimum_pad_normal_force_n=0.):
    """Sampled load feasibility for both correction signs, with no support.

    The requested moment is I*alpha for the actual controller's acceleration
    bound. The restricted points and inner friction cone are not an outer
    relaxation: failure is not proof of physical impossibility. Passing does
    not establish frictional sticking or closed-loop gain.
    """
    obj=meta['object'];R=pin.Quaternion(q[obj['q']+3:obj['q']+7]).matrix()
    inertia=R@np.asarray(obj['inertia'])@R.T;results=[]
    for axis in np.eye(3):
        for sign in [-1.,1.]:
            moment=sign*angular_acceleration*(inertia@axis)
            result=static_balance(model,meta,q,contacts,object_wrench=np.r_[np.zeros(3),moment],minimum_pad_normal_force_n=minimum_pad_normal_force_n)
            results.append(dict(axis=axis.tolist(),sign=sign,required_moment_nm=moment.tolist(),
                status=result['status'],max_normalized_joint_effort=result.get('max_normalized_joint_effort')))
    return dict(scope='sampled_static_restoring_load_screen',passed=all(r['status']=='static_screen_pass' for r in results),directions=results)


def refine_gimbal_trim(model,meta,q,contacts,initial,*,minimum_normal=.5):
    """Use the force LP's envelope gradient, accepting only freshly solved trims."""
    q=q.copy();indices=np.array([meta['actuators'][r['gimbal']]['q']for r in meta['rotors']])
    low=np.array([meta['actuators'][r['gimbal']]['lower']+.02 for r in meta['rotors']])
    high=np.array([meta['actuators'][r['gimbal']]['upper']-.02 for r in meta['rotors']])
    records=[];current=initial
    for iteration in range(12):
        current=static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=minimum_normal,gimbal_gradient=True)
        peak=current.get('minimum_peak_effort_ratio',np.inf)
        records.append(dict(peak_effort_ratio=peak))
        if peak<=.95 or 'gimbal_peak_gradient' not in current:break
        gradient=np.array(current['gimbal_peak_gradient']);gradient/=max(abs(gradient).max(),1e-12)
        for step in [.3,.15,.075,.0375]:
            moved=q.copy();moved[indices]=np.clip(q[indices]-step*gradient,low,high)
            trial=static_balance(model,meta,moved,contacts,minimum_pad_normal_force_n=minimum_normal)
            if trial.get('minimum_peak_effort_ratio',np.inf)<peak-1e-5:
                q,current=moved,trial;break
        else:break
    current.pop('gimbal_peak_gradient',None)
    current['gimbal_refinement']=records
    return q,current
