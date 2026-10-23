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


def static_balance(model,meta,q,contacts=(),time_limit_s=None, *, payload_fraction=1., balance_thrust=False,object_wrench=None,base_load=None,base_acceleration=None,peak_slack=None,minimum_pad_normal_force_n=0.):
    """Contacts give force on the robot; the opposite force acts on the object.

    normal, point and target_point are expressed in world/each local frame as
    indicated. A diamond inner friction cone and l1 force cap are conservative.
    Failure of this screen does not prove physical infeasibility.
    """
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
    columns=[];bounds=[];contact_columns=[]
    for rotor in meta['rotors']:
        frame=rotor['frame'];J=pin.computeFrameJacobian(model,data,q,frame,pin.LOCAL_WORLD_ALIGNED)
        columns.append(J[:3].T@data.oMf[frame].rotation[:,2]);bounds.append((rotor['minimum'],rotor['maximum']))
    for c in contacts:
        n=np.asarray(c['normal'],dtype=float)
        if abs(np.linalg.norm(n)-1)>1e-9:raise ValueError('non-unit normal')
        axis=np.eye(3)[np.argmin(abs(n))];t1=np.cross(n,axis);t1/=np.linalg.norm(t1);t2=np.cross(n,t1)
        R=np.column_stack([n,t1,t2]);J=point_jacobian(model,data,q,meta['frames'][c['frame']],c['point'])
        if c.get('target_frame'):
            J-=point_jacobian(model,data,q,meta['frames'][c['target_frame']],c['target_point'])
        first=len(columns);contact_columns.append((first,c,R))
        columns.extend((J.T@R)[:,i] for i in range(3))
        bounds.extend([(0,c['max_force']), (None,None),(None,None)])
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
    for a in meta['actuators']:
        row=np.zeros(nx);row[:-1]=-B[a['v']];row[-1]=-a['effort']
        A.append(row);b.append(-gravity[a['v']])
        row=row.copy();row[:-1]*=-1;A.append(row);b.append(gravity[a['v']])
    for first,c,R in contact_columns:
        for sx in [-1,1]:
            for sy in [-1,1]:
                row=np.zeros(nx);row[first:first+3]=[-c['mu'],sx,sy];A.append(row);b.append(0.)
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
    objective=np.zeros(nx);objective[-1]=1.
    objective[:len(meta['rotors'])]=1e-8
    options={'maxiter':500}
    if time_limit_s is not None:options['time_limit']=float(time_limit_s)
    solution=linprog(objective,A_ub=np.array(A),b_ub=np.array(b),A_eq=E,b_eq=gravity[unactuated],bounds=bounds+[(0,None)],
                     method='highs',options=options)
    result=dict(scope='fixed_pose_inner_force_screen_only',task_accepted=False,
                solver_status=int(solution.status),solver_message=solution.message,iterations=int(solution.nit),
                wall_s=time.perf_counter()-start,physical_impossibility_proven=False)
    if not solution.success:
        result['status']='unresolved_or_inner_model_infeasible';return result
    x=solution.x
    result['minimum_peak_effort_ratio']=float(x[-1])
    if balance_thrust:
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
    torque=gravity-B@x[:-1]
    effort_ratio=max(abs(float(torque[a['v']]))/a['effort'] for a in meta['actuators'])
    residual=float(np.max(abs(torque[unactuated]),initial=0.))
    violation=float(np.max(np.array(A)@x-np.array(b),initial=0.))
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
