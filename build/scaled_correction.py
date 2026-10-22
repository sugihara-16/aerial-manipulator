"""Largest attainable correction followed by the original quadratic objective.

The variables are offsets from an unchanged supporting input. This contracts
only the requested correction, never gravity support or the complete input.
"""
import numpy as np
import scipy.sparse as sp
import clarabel
from amsrr_det.linear_program import linprog
from build.allocation_optimality import box_objective_gap


def scaled_correction(H, g, feedback_gradient, A, b, lower, upper):
    n=len(lower)
    ray=linprog(np.r_[np.zeros(n),-1.],A_eq=np.c_[A,-b],b_eq=np.zeros(len(b)),
        bounds=list(zip(lower,upper))+[(0.,1.)],
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not ray.success:
        raise ValueError('no attainable correction scale')
    alpha=float(np.clip(ray.x[-1],0.,1.));target=alpha*b
    gradient=g+(alpha-1.)*feedback_gradient
    U,s,V=np.linalg.svd(A,full_matrices=True)
    rank=np.count_nonzero(s>s[0]*1e-12) if len(s) else 0
    origin=ray.x[:n].copy()
    origin+=V[:rank].T@((U[:,:rank].T@(target-A@origin))/s[:rank])
    N=V[rank:].T
    if N.shape[1]:
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-11
        answer=clarabel.DefaultSolver(sp.csc_matrix(np.triu(N.T@H@N)),N.T@(H@origin+gradient),
            sp.csc_matrix(np.r_[N,-N]),np.r_[upper-origin,origin-lower],
            [clarabel.NonnegativeConeT(2*n)],settings).solve()
        if str(answer.status)!='Solved':
            raise ValueError('scaled correction objective unresolved: '+str(answer.status))
        x=origin+N@np.asarray(answer.x)
        box_dual=np.asarray(answer.z)[:n]-np.asarray(answer.z)[n:]
    else:
        x=origin;box_dual=np.zeros(n)
    equality_dual=np.linalg.lstsq(A.T,-(H@x+gradient+box_dual),rcond=1e-12)[0]
    objective_gap=box_objective_gap(H,gradient,A,target,lower,upper,x,equality_dual)
    residual=max(float(np.max(abs(A@x-target),initial=0.)),float(np.max(lower-x)),float(np.max(x-upper)))
    if not np.isfinite(residual+objective_gap)or residual>1e-7 or objective_gap>1e-7:
        raise ValueError('scaled correction residual: '+str((residual,objective_gap)))
    return x,alpha,dict(normalized_objective_gap_bound=objective_gap,normalized_residual=residual)
