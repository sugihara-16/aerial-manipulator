"""Local quasistatic preload with the authored finite friction relation.

This diagnostic couples drive effort, compression and a small relative drift.
Its fixed-geometry, constant-rate bounds are not a trajectory certificate.
The evaluator's forces and completed simulation outcomes are never inputs.
"""
import numpy as np
import scipy.linalg as la
import quadprog
import scipy.sparse as sp
from scipy.optimize import minimize,LinearConstraint
from amsrr_det.linear_program import linprog
import clarabel, mujoco
from build.momentum_feedback import rigid_motion_map
from build.contact_tangent import constraint_jacobian


def _dense_pose(H,g,A,b):
    try:
        return _whitened_pose(H,g,A,b)
    except (ValueError,np.linalg.LinAlgError):
        # Whitening can magnify nearly dependent constraint rows. Solve the
        # identical positive-definite QP in its original coordinates before
        # falling back to the slower general optimizer.
        x,_,_,_,dual,_=quadprog.solve_qp(H,-g,-A.T,-b)
        dual=np.maximum(dual,0.)
        gradient=H@x+g+A.T@dual
        gap=.5*gradient@np.linalg.solve(H,gradient)+dual@np.maximum(b-A@x,0.)
        # The reduced objective omits a large constant at the feasible origin.
        # Include relative roundoff in that shifted objective, while preserving
        # the original constraint tolerance and downstream physical recheck.
        tolerance=1e-9+1e-12*(abs(x@(H@x))+abs(g@x))
        if not np.isfinite(x).all() or np.max(A@x-b,initial=0.)>1e-8 or gap>tolerance:
            raise ValueError('dense QP physical residual or objective error')
        return x


def _whitened_pose(H,g,A,b):
    factor=la.cholesky(H,lower=True,check_finite=False)
    transform=la.solve_triangular(factor,np.eye(len(g)),lower=True,check_finite=False).T
    linear=transform.T@g;matrix=A@transform
    row=np.maximum(np.linalg.norm(matrix,axis=1),1e-12)
    C=matrix/row[:,None];limit=b/row
    answer,_,_,_,dual,_=quadprog.solve_qp(np.eye(len(g)),-linear,-C.T,-limit,factorized=True)
    dual=np.maximum(dual,0.)
    stationarity=answer+linear+C.T@dual
    gap=.5*stationarity@stationarity+dual@np.maximum(limit-C@answer,0.)
    z=transform@answer
    if not np.isfinite(z).all() or np.max(A@z-b,initial=0.)>1e-8 or gap>1e-9:
        raise ValueError('dense QP physical residual or objective error')
    return z


def finite_slip_preload(local, q, duration, *, move_pose=False, travel_budget=.03,minimize_input=False):
    local.at(q)
    gap, tangent, Jn, Ct = local.contact()
    mass, act, _ = local.matrices()
    m = local.m; nv, nu, nj = m.nv, m.nu, local.nj
    if duration <= 0 or local.support:
        raise ValueError('positive airborne duration required')
    if not local.elements:
        return None,dict(feasible=False,message='no active robot-object contacts')
    if not 0.<travel_budget<=.03:
        return None,dict(feasible=False,message='remaining material travel exhausted')
    # Variables: displacement, actual effort/thrust, velocity, shared reserve.
    width = 2*nv+nu+1; vs = slice(nv+nu, width-1)
    B = act/np.r_[local.gain, np.ones(nu-nj)]
    # Obtain coefficients of the authored discrete contact law at this pose.
    # The query is private, has zero velocity, and is not advanced in time.
    # No computed reaction is read: only geometry and constitutive coefficients.
    local.d.ctrl[:]=np.r_[np.where(local.dock,local.kp*q[local.qi],q[local.qi]),np.zeros(nu-nj)]
    mujoco.mj_forward(m,local.d)
    contact_rows=[]
    for contact in local.d.contact:
        if contact.dist>1e-7 or contact.efc_address<0:continue
        ga,gb=map(int,contact.geom);oid=m.geom('object').id
        if oid not in (ga,gb):continue
        other=gb if ga==oid else ga
        if any(m.geom(other).name.startswith(link+'/')for link in local.links):
            contact_rows.append(int(contact.efc_address))
    if len(contact_rows)!=len(local.elements):raise ValueError('local contact correspondence changed')
    addresses=np.array(contact_rows,dtype=int)[:,None]+np.arange(3)
    actual_J=constraint_jacobian(m,local.d)[addresses]
    Jn=actual_J[:,0];Ct=actual_J[:,1:].reshape(-1,nv)
    kbip=local.d.efc_KBIP[addresses];h=m.opt.timestep
    spring=kbip[:,:,0]*kbip[:,:,2];damping=kbip[:,:,1]
    coefficient=local.d.efc_D[addresses]/(1+h*damping+h*h*spring)
    kn=coefficient[:,0]*spring[:,0]
    cn=coefficient[:,0]*(damping[:,0]+h*spring[:,0])
    ct=(coefficient[:,1:]*damping[:,1:]).reshape(-1)
    gap=local.d.efc_pos[addresses[:,0]]-local.d.efc_margin[addresses[:,0]]
    pressure = np.zeros((len(gap), width))
    pressure[:, :nv] = -kn[:,None]*Jn
    pressure[:, vs] = -cn[:,None]*Jn
    pressure_end = pressure.copy(); pressure_end[:, vs] -= duration*kn[:,None]*Jn
    p0 = -kn*gap
    tangent_force = np.zeros((len(Ct), width)); tangent_force[:, vs] = -ct[:,None]*Ct
    balance = Jn.T@pressure+Ct.T@tangent_force
    balance[:, nv:nv+nu] = B
    common = np.zeros((6, width)); common[:, vs] = rigid_motion_map(m,q).T@mass
    E = np.vstack([balance, common]); rhs = np.r_[local.d.qfrc_bias-Jn.T@p0, np.zeros(6)]
    rows=[]; bounds=[]
    def upper(row, value):
        rows.append(np.atleast_2d(row)); bounds.append(np.broadcast_to(value,(len(np.atleast_2d(row)),)))
    pads=np.array([e[0] for e in local.elements])
    lever=np.array([np.linalg.norm(e[3]) for e in local.elements])
    directions=np.c_[np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)]
    for pmap in (pressure, pressure_end):
        upper(-pmap,p0)
        for i in range(len(gap)):
            friction=directions@tangent_force[2*i:2*i+2]-4.5*np.cos(np.pi/8)*pmap[i]
            friction[:,-1]=4.5*np.cos(np.pi/8)*.5/np.count_nonzero(pads==pads[i])
            upper(friction,4.5*np.cos(np.pi/8)*p0[i])
        for pad in range(len(local.links)):
            selected=pads==pad; total=pmap[selected].sum(axis=0)
            minimum=-total.copy();minimum[-1]=.5
            upper(minimum,p0[selected].sum()-.5)
            factor=np.sqrt(1+4.5**2)
            upper(factor*total,30-factor*p0[selected].sum())
            upper(factor*lever[selected]@pmap[selected],5-factor*lever[selected]@p0[selected])
    limits=np.r_[local.effort,local.rotor_max]
    effort=np.zeros((nu,width));effort[:,nv:nv+nu]=np.eye(nu)
    for sign in (-1,1):
        constraint=sign*effort;constraint[:,-1]=limits
        upper(constraint,limits if sign==1 else np.r_[local.effort,-local.rotor_min])
        issuance=sign*effort[:nj].copy()
        issuance[np.arange(nj),nv+nu+local.vi]+=sign*h*local.kp
        issuance[:,-1]=local.effort
        upper(issuance,local.effort)
        # A held command's endpoint torque includes accumulated servo error.
        endpoint=sign*effort[:nj].copy()
        endpoint[np.arange(nj),nv+nu+local.vi]-=sign*duration*local.kp
        endpoint[:,-1]=local.effort
        upper(endpoint,local.effort)
    travel=np.zeros((len(Ct),width));travel[:,:nv]=Ct;travel[:,vs]=duration*Ct
    upper(travel,travel_budget/np.sqrt(2));upper(-travel,travel_budget/np.sqrt(2))
    joint_end=np.zeros((nj,width));joint_end[np.arange(nj),local.vi]=1.
    joint_end[np.arange(nj),nv+nu+local.vi]=duration
    upper(joint_end,m.jnt_range[local.joint_ids,1]-q[local.qi])
    upper(-joint_end,q[local.qi]-m.jnt_range[local.joint_ids,0])
    command=np.zeros((nu,width));command[:,nv:nv+nu]=np.diag(np.r_[1/local.gain,np.ones(nu-nj)])
    command[np.arange(nj),local.vi]=local.kp/local.gain
    command[np.arange(nj),nv+nu+local.vi]=(local.kd+h*local.kp)/local.gain
    command0=np.r_[local.kp*q[local.qi]/local.gain,np.zeros(nu-nj)]
    limited=m.actuator_ctrllimited.astype(bool)
    upper(command[limited],m.actuator_ctrlrange[limited,1]-command0[limited])
    upper(-command[limited],command0[limited]-m.actuator_ctrlrange[limited,0])
    gimbal=np.flatnonzero(~local.dock)
    upper(command[gimbal],m.jnt_range[local.joint_ids[gimbal],1]-command0[gimbal])
    upper(-command[gimbal],command0[gimbal]-m.jnt_range[local.joint_ids[gimbal],0])
    trust=np.full(nv,.005 if move_pose else 0.)
    speed=np.full(nv,3.)
    for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
        vi=int(m.jnt_dofadr[j]);trust[vi:vi+3]=.001 if move_pose else 0.
        speed[vi:vi+6]=np.r_[[.03]*3,[.2]*3]/(duration*np.sqrt(3))
    oi=int(m.jnt_dofadr[m.joint('object').id]);trust[oi:oi+6]=0.
    trust[local.vi[~local.dock]]=0.
    speed[local.vi]=np.minimum(3.,2*local.effort/local.kp/duration)
    speed[local.vi[~local.dock]]=0.
    lo=np.r_[-trust,-limits,-speed,0.];hi=np.r_[trust,limits,speed,.5]
    lo[nv+nj:nv+nu]=local.rotor_min
    lo[local.vi]=np.maximum(lo[local.vi],m.jnt_range[local.joint_ids,0]-q[local.qi])
    hi[local.vi]=np.minimum(hi[local.vi],m.jnt_range[local.joint_ids,1]-q[local.qi])
    # Use a force-scale velocity for numerical conditioning of viscous friction.
    scale=np.r_[np.maximum(trust,.001),limits,np.full(nv,1/np.max(ct)),1.]
    A=np.vstack(rows)*scale;b=np.concatenate(bounds);eq=E*scale
    norm=np.maximum(np.linalg.norm(eq,axis=1),1e-12)
    objective=np.zeros(width);objective[-1]=-1.
    first=linprog(objective,A_ub=A,b_ub=b,A_eq=eq/norm[:,None],b_eq=rhs/norm,
        bounds=list(zip(lo/scale,hi/scale)),options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not first.success:return None,dict(feasible=False,message=first.message)
    lo[-1]=.95*first.x[-1]
    def secondary():
        fixed=lo==hi;free=~fixed;origin=np.where(fixed,lo/scale,0.)
        reduced_eq=eq[:,free];reduced_rhs=rhs-eq@origin
        reduced_eq=reduced_eq/norm[:,None];reduced_rhs=reduced_rhs/norm
        U,s,V=np.linalg.svd(reduced_eq,full_matrices=True);rank=np.count_nonzero(s>s[0]*1e-14)
        # Optimize only directions that preserve force/momentum balance. Keeping
        # these stiff equalities inside the secondary QP allowed an approximate
        # solver exit to violate the original physical residual on valid LPs.
        origin[free]=first.x[free]+V[:rank].T@((U[:,:rank].T@reduced_rhs)/s[:rank]-V[:rank]@first.x[free])
        null=V[rank:].T
        identity=np.eye(np.count_nonzero(free));upper_ids=np.isfinite(hi[free]);lower_ids=np.isfinite(lo[free])
        matrix=np.vstack([A[:,free],identity[upper_ids],-identity[lower_ids]])
        bound=np.r_[b-A@origin,(hi/scale)[free][upper_ids]-origin[free][upper_ids],
                     -(lo/scale)[free][lower_ids]+origin[free][lower_ids]]
        weights=np.r_[np.ones(nv),np.full(nu,1e-4),np.ones(nv),1e-8]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-11
        normalized=origin.copy()
        if null.shape[1]:
            objective=null.T@(weights[free,None]*null);linear=null.T@(weights[free]*origin[free])
            constraints=matrix@null;solution=None
            if move_pose:
                try:solution=_dense_pose(objective,linear,constraints,bound)
                except (ValueError,np.linalg.LinAlgError):pass
            if move_pose and solution is None:
                # The LP origin is feasible. A dense active-set solve avoids the
                # large sparse factorization of these dense null-space rows.
                row_scale=np.maximum(np.linalg.norm(constraints,axis=1),1.)
                result=minimize(lambda z:(.5*z@objective@z+linear@z,objective@z+linear),
                    np.zeros(null.shape[1]),jac=True,method='SLSQP',
                    constraints=[LinearConstraint(constraints/row_scale[:,None],-np.inf,bound/row_scale)],
                    options=dict(maxiter=100,ftol=1e-11))
                if result.success and np.max(constraints@result.x-bound)<1e-8:
                    solution=result.x
            if solution is None:
                result=clarabel.DefaultSolver(sp.csc_matrix(np.triu(objective)),linear,sp.csc_matrix(constraints),bound,
                    [clarabel.NonnegativeConeT(len(bound))],settings).solve()
                if str(result.status) not in ['Solved','AlmostSolved']:raise ValueError(str(result.status))
                solution=np.asarray(result.x)
            normalized[free]+=null@solution
        return normalized
    try:
        normalized=first.x.copy() if minimize_input and not move_pose else secondary()
    except ValueError as error:return None,dict(feasible=False,message=str(error))
    input_refinement=None
    if minimize_input and not move_pose:
        from build.minimum_input_lp import minimum_input
        indices=np.arange(nv+nj,nv+nu)
        problem=dict(A_ub=A,b_ub=b,A_eq=eq/norm[:,None],b_eq=rhs/norm,
                     bounds=list(zip(lo/scale,hi/scale)))
        # The power-cone formulation solves the same convex thrust objective
        # directly, avoiding a piecewise LP followed by dense SLSQP iterations.
        # Try the final thrust objective directly from the feasible reserve LP.
        # The intermediate quadratic objective is only a numerical fallback.
        for attempt in range(2):
            refined,input_refinement=minimum_input(problem,normalized,indices,local.rotor_max,lo[-1],method='conic')
            # Keep the original physical-unit check, beyond the normalized solve.
            candidate=refined*scale
            physical_residual=max(float(abs(E@candidate-rhs).max()),float((A@refined-b).max()),
                                  float((lo-candidate).max()),float((candidate-hi).max()))
            input_refinement['physical_residual']=physical_residual
            input_refinement['used']=bool(input_refinement['accepted'] and physical_residual<1e-6)
            if input_refinement['used']:
                normalized=refined;break
            if attempt==0:
                try:normalized=secondary()
                except ValueError as error:return None,dict(feasible=False,message=str(error))
    x=normalized*scale
    residual=max(float(abs(E@x-rhs).max()),float((A@(x/scale)-b).max()),float((lo-x).max()),float((x-hi).max()))
    updated=q.copy();mujoco.mj_integratePos(m,updated,x[:nv],1.)
    v=x[vs];u=x[nv:nv+nu].copy()
    u[:nj]=(u[:nj]+local.kp*updated[local.qi]+(local.kd+h*local.kp)*v[local.vi])/local.gain
    info=dict(feasible=residual<1e-6,residual=residual,reserve=float(first.x[-1]),
        displacement_max=float(abs(x[:nv]).max()),velocity_max=float(abs(v).max()),
        normal_rate_max=float(abs(Jn@v).max()),joint_rate_max=float(abs(v[local.vi]).max()),
        tangent_law_residual=float(abs(tangent_force@x+ct*(Ct@v)).max()),duration_s=duration,material_travel_budget_m=travel_budget)
    if input_refinement is not None:info['input_refinement']=input_refinement
    return (updated,u,v),info


def construct_finite_slip(plant,full_model,q,links,duration, *, minimize_input=False,travel_budget=.03,clearance=None):
    """Refine geometry, then solve the same law again at the actual pose."""
    from build.sticking_design import LocalGrasp
    from amsrr_det.clearance import MotionBoundedClearance
    clearance=clearance or MotionBoundedClearance(full_model,links);records=[]
    fixed,check=finite_slip_preload(LocalGrasp(plant,q,links),q,duration,move_pose=False,minimize_input=minimize_input,travel_budget=travel_budget)
    if fixed is not None and check['feasible']:
        if clearance.check(q,True)[0]:raise ValueError('finite-slip initial pose clearance')
        return fixed,dict(scope=__doc__,initial_fixed=check,iterations=records)
    for iteration in range(12):
        local=LocalGrasp(plant,q,links)
        answer,step=finite_slip_preload(local,q,duration,move_pose=True,travel_budget=travel_budget)
        if answer is None or not step['feasible']:raise ValueError('finite-slip pose '+str(step))
        q=answer[0]
        if clearance.check(q,True)[0]:raise ValueError('finite-slip pose clearance')
        local=LocalGrasp(plant,q,links)
        fixed,check=finite_slip_preload(local,q,duration,move_pose=False,minimize_input=minimize_input,travel_budget=travel_budget)
        records.append(dict(iteration=iteration,pose=step,recheck=check))
        if fixed is not None and check['feasible']:
            return fixed,dict(scope=__doc__,iterations=records)
    raise ValueError('finite-slip geometry refinement did not converge')
