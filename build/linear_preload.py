"""Local preload step from force balance, material geometry and input limits.

The closest pose with a shared force/input reserve is obtained by a linear
program followed by a convex quadratic program. No dynamics rollout is used.
The frozen geometry is re-evaluated outside this local step.
"""
import numpy as np
from amsrr_det.linear_program import linprog
import scipy.sparse as sp
import clarabel
import mujoco


def preload_step(local,q,*,fixed_material=True,acceleration=None):
    local.at(q);gap,tangent,Jn,Ct=local.contact();mass,act,_=local.matrices()
    m=local.m;nv=m.nv;nu=m.nu;nt=len(Ct);nj=local.nj;width=nv+nu+nt+1
    B=act/np.r_[local.gain,np.ones(nu-nj)]
    pads=np.array([e[0]for e in local.elements]);p0=-local.k*gap
    pmap=np.zeros((len(gap),width));pmap[:,:nv]=-local.k*Jn
    eq=np.zeros((nv+(nt if fixed_material else 0),width));eq[:nv,:nv]=-local.k*Jn.T@Jn
    eq[:nv,nv:nv+nu]=B;eq[:nv,nv+nu:-1]=Ct.T
    if fixed_material:eq[nv:,:nv]=Ct
    rhs=np.r_[local.d.qfrc_bias-Jn.T@p0,np.zeros(nt if fixed_material else 0)]
    if acceleration is not None:rhs[:nv]+=mass@np.asarray(acceleration)
    rows=[];bounds=[]
    if not fixed_material:
        movement=np.zeros((nt,width));movement[:,:nv]=Ct
        rows.extend(movement);rows.extend(-movement);bounds.extend([.03/np.sqrt(2)]*(2*nt))
    # Nonnegative pressure at each actual element.
    rows.extend(-pmap);bounds.extend(p0)
    angles=np.arange(8)*np.pi/4;directions=np.c_[np.cos(angles),np.sin(angles)]
    mu=4.5*np.cos(np.pi/8)
    for k in range(len(gap)):
        for direction in directions:
            row=-mu*pmap[k].copy();row[nv+nu+2*k:nv+nu+2*k+2]=direction
            row[-1]=mu*.5/np.count_nonzero(pads==pads[k]);rows.append(row);bounds.append(mu*p0[k])
    for pad in range(len(local.links)):
        selected=pads==pad;row=-pmap[selected].sum(axis=0);row[-1]=.5
        rows.append(row);bounds.append(p0[selected].sum()-.5)
        # Triangle inequalities imply the complete-pad force/moment bounds.
        factor=np.sqrt(1+4.5**2);row=factor*pmap[selected].sum(axis=0)
        rows.append(row);bounds.append(30.-factor*p0[selected].sum())
        lever=np.array([np.linalg.norm(e[3])for e in local.elements])[selected]
        row=factor*lever@pmap[selected];rows.append(row);bounds.append(5.-factor*lever@p0[selected])
    limits=np.r_[local.effort,local.rotor_max]
    for i,limit in enumerate(limits):
        row=np.zeros(width);row[nv+i]=1.;row[-1]=limit;rows.append(row);bounds.append(limit)
        row=row.copy();row[nv+i]=-1.;rows.append(row);bounds.append(limit if i<nj else 0.)
    trust=np.full(nv,.005)
    for j in np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE):
        vi=m.jnt_dofadr[j];trust[vi:vi+3]=.001
    obj=m.jnt_dofadr[m.joint('object').id];trust[obj:obj+6]=0.
    trust[local.vi[~local.dock]]=0.
    lo=np.r_[-trust,-limits,np.full(nt,-np.inf),0.];hi=np.r_[trust,limits,np.full(nt,np.inf),.5]
    lo[nv+nj:nv+nu]=0.
    lo[local.vi]=np.maximum(lo[local.vi],m.jnt_range[local.joint_ids,0]-q[local.qi])
    hi[local.vi]=np.minimum(hi[local.vi],m.jnt_range[local.joint_ids,1]-q[local.qi])
    scale=np.r_[np.maximum(trust,.001),limits,np.ones(nt),1.]
    A=np.array(rows)*scale;bound=np.array(bounds);E=eq*scale
    objective=np.zeros(width);objective[-1]=-1.
    row_scale=np.maximum(np.linalg.norm(E,axis=1),1e-12)
    first=linprog(objective,A_ub=A,b_ub=bound,A_eq=E/row_scale[:,None],b_eq=rhs/row_scale,
                  bounds=list(zip(lo/scale,hi/scale)),options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not first.success:raise ValueError('local preload: '+first.message)
    lo[-1]=.95*first.x[-1]
    # Scale the independent equilibrium equations without changing their span.
    U,S,V=np.linalg.svd(E,full_matrices=False);rank=np.count_nonzero(S>S[0]*1e-11)
    equality=V[:rank];value=(U[:,:rank].T@rhs)/S[:rank]
    lower=lo/scale;upper=hi/scale;identity=np.eye(width)
    upper_ids=np.isfinite(upper);lower_ids=np.isfinite(lower)
    matrix=np.vstack([equality,A,identity[upper_ids],-identity[lower_ids]])
    b=np.r_[value,bound,upper[upper_ids],-lower[lower_ids]]
    weights=np.r_[np.ones(nv),np.full(nu,1e-4),np.full(nt,1e-6),1e-8]
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
    solver=clarabel.DefaultSolver(sp.diags(weights,format='csc'),np.zeros(width),sp.csc_matrix(matrix),b,
        [clarabel.ZeroConeT(rank),clarabel.NonnegativeConeT(len(b)-rank)],settings)
    result=solver.solve()
    if result.status not in [clarabel.SolverStatus.Solved,clarabel.SolverStatus.AlmostSolved]:raise ValueError('local preload '+str(result.status))
    x=np.array(result.x)*scale;pressure=p0+pmap@x
    residual=max(float(abs(eq@x-rhs).max()),float((np.array(rows)@x-bound).max()),float((lo-x).max()),float((x-hi).max()))
    if residual>1e-6:raise ValueError('local preload residual '+str(residual))
    updated=q.copy();mujoco.mj_integratePos(m,updated,x[:nv],1.)
    return updated,dict(reserve=float(first.x[-1]),retained_reserve=float(x[-1]),constraint_error=residual,
                        fixed_material=fixed_material,linearized_tangent_displacement_m=float(np.max(np.linalg.norm((Ct@x[:nv]).reshape(-1,2),axis=1))),
                        predicted_loads=[float(sum(pressure[pads==pad]))for pad in range(len(local.links))],
                        scaled_displacement=float(max(abs(x[:nv])/np.maximum(trust,.001))))
