"""Minimize thrust**1.5 inside a fixed, already-feasible linear design model.

No posture, geometry, friction, force, or margin constraints are removed. This
is a convex input refinement, not a certificate over other robot structures.
Power cone definition: https://clarabel.org/stable/api_cone_types/
"""
import time
import numpy as np
import scipy.sparse as sp
import clarabel
from scipy.optimize import minimize,LinearConstraint
from amsrr_det.linear_program import linprog


def minimum_input(problem, feasible, rotor_columns, rotor_scales, minimum_reserve, *, method='hybrid'):
    started=time.perf_counter()
    A=np.asarray(problem['A_ub']);b=np.asarray(problem['b_ub'])
    E=np.asarray(problem['A_eq']);rhs=np.asarray(problem['b_eq'])
    lo=np.array([-np.inf if l is None else l for l,h in problem['bounds']])
    hi=np.array([np.inf if h is None else h for l,h in problem['bounds']])
    if method not in ('hybrid','slsqp','conic','piecewise'):
        raise ValueError('unknown input refinement method')
    if (np.any(lo[rotor_columns]<0) or np.any(hi[rotor_columns]>1)
            or not np.isfinite(rotor_scales).all() or np.any(np.asarray(rotor_scales)<=0)):
        raise ValueError('rotor coordinates must be normalized to [0,1] with positive scales')
    lo[-1]=max(lo[-1],minimum_reserve)
    fixed=lo==hi;free=~fixed;origin=np.where(fixed,lo,0.)
    U,s,V=np.linalg.svd(E[:,free],full_matrices=True)
    rank=np.count_nonzero(s>s[0]*1e-14)
    origin[free]=feasible[free]+V[:rank].T@((U[:,:rank].T@(rhs-E@origin))/s[:rank]-V[:rank]@feasible[free])
    null=np.zeros((len(origin),len(V)-rank));null[free]=V[rank:].T
    nz=null.shape[1];nr=len(rotor_columns);width=nz+nr
    ids_hi=np.flatnonzero(np.isfinite(hi));ids_lo=np.flatnonzero(np.isfinite(lo))
    C=np.vstack([A,np.eye(len(lo))[ids_hi],-np.eye(len(lo))[ids_lo]])
    d=np.r_[b,hi[ids_hi],-lo[ids_lo]]
    matrix=np.c_[C@null,np.zeros((len(C),nr))];bound=d-C@origin
    initial_z=np.zeros(nz)
    if method in ('piecewise','hybrid'):
        # Convex secants on [0,1] bound normalized f**1.5 from above.
        # Their worst interpolation error is explicit, including near zero.
        knots=np.linspace(0.,1.,17)**2
        slopes=np.diff(knots**1.5)/np.diff(knots)
        intercepts=knots[:-1]**1.5-slopes*knots[:-1]
        critical=(slopes/1.5)**2
        error=float(np.max(slopes*critical+intercepts-critical**1.5))
        rows=np.zeros((nr*len(slopes),width));bounds=np.zeros(len(rows))
        for i,col in enumerate(rotor_columns):
            start=i*len(slopes);end=start+len(slopes)
            rows[start:end,:nz]=slopes[:,None]*null[col]
            rows[start:end,nz+i]=-1.
            bounds[start:end]=-intercepts-slopes*origin[col]
        weights=np.asarray(rotor_scales)**1.5
        answer=linprog(np.r_[np.zeros(nz),weights/weights.max()],
            A_ub=sp.csr_matrix(np.vstack([matrix,rows])),b_ub=np.r_[bound,bounds],
            bounds=[(None,None)]*nz+[(0,None)]*nr,
            options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10,
                         small_matrix_value=1e-12))
        x=origin+null@answer.x[:nz] if answer.success else feasible.copy()
        residual=max(float(abs(E@x-rhs).max()),float((C@x-d).max()))
        force=np.maximum(0,x[rotor_columns]*rotor_scales)
        cost=float(np.sum(force**1.5));gap=error*weights.sum()
        lower=float(answer.fun*weights.max()-gap) if answer.success else None
        report=dict(status=answer.message,normalized_residual=residual,burden=cost,
                    initial_burden=float(np.sum((feasible[rotor_columns]*rotor_scales)**1.5)),
                    interpolation_error_bound=float(gap),model_optimum_lower_bound=lower,
                    minimum_reserve=minimum_reserve,actual_reserve=float(x[-1]),
                    wall_s=time.perf_counter()-started,iterations=answer.nit)
        report['accepted']=(answer.success and residual<1e-8 and cost<=report['initial_burden']+1e-5)
        if method=='piecewise':return x,report
        if answer.success:initial_z=answer.x[:nz]
    if method in ('slsqp','hybrid'):
        force0=origin[rotor_columns]*rotor_scales
        derivative=null[rotor_columns]*np.asarray(rotor_scales)[:,None]
        def cost(z):
            force=np.maximum(0,force0+derivative@z)
            return float(np.sum(force**1.5)),1.5*np.sqrt(force)@derivative
        scale=np.maximum(np.linalg.norm(matrix[:,:nz],axis=1),1.)
        result=minimize(cost,initial_z,jac=True,method='SLSQP',
                        constraints=[LinearConstraint(matrix[:,:nz]/scale[:,None],-np.inf,bound/scale)],
                        options=dict(maxiter=100,ftol=1e-9))
        x=origin+null@result.x
        residual=max(float(abs(E@x-rhs).max()),float((C@x-d).max()))
        report=dict(status=str(result.message),normalized_residual=residual,
                    burden=cost(result.x)[0],initial_burden=float(np.sum((feasible[rotor_columns]*rotor_scales)**1.5)),
                    minimum_reserve=minimum_reserve,actual_reserve=float(x[-1]),
                    wall_s=time.perf_counter()-started,iterations=result.nit)
        report['accepted']=(residual<1e-8 and report['burden']<=report['initial_burden']+1e-5)
        return x,report
    blocks=[matrix];values=[bound];cones=[clarabel.NonnegativeConeT(len(bound))]
    for i,col in enumerate(rotor_columns):
        # (t_i,1,normalized_thrust_i) in power cone with exponent 2/3.
        block=np.zeros((3,width));block[0,nz+i]=-1.;block[2,:nz]=-null[col]
        blocks.append(block);values.append(np.array([0.,1.,origin[col]]))
        cones.append(clarabel.PowerConeT(2/3))
    objective=np.r_[np.zeros(nz),np.asarray(rotor_scales)**1.5]
    objective/=np.max(objective)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-11
    answer=clarabel.DefaultSolver(sp.csc_matrix((width,width)),objective,
        sp.csc_matrix(np.vstack(blocks)),np.concatenate(values),cones,settings).solve()
    result=origin+null@np.asarray(answer.x)[:nz]
    residual=max(float(abs(E@result-rhs).max()),float((C@result-d).max()))
    force=result[rotor_columns]*rotor_scales
    old_force=feasible[rotor_columns]*rotor_scales
    report=dict(status=str(answer.status),normalized_residual=residual,
                burden=float(np.sum(np.maximum(0,force)**1.5)),
                initial_burden=float(np.sum(np.maximum(0,old_force)**1.5)),
                minimum_reserve=minimum_reserve,actual_reserve=float(result[-1]),
                wall_s=time.perf_counter()-started,iterations=answer.iterations)
    report['accepted']=(str(answer.status) in ['Solved','AlmostSolved'] and residual<1e-8
                        and np.min(force)>=-1e-8 and report['burden']<=report['initial_burden']+1e-5)
    return result,report
