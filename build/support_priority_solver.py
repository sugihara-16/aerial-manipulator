"""Minimize current finite support load before the existing tracking cost.

Previous predicted progress is not a physical constraint. This is a local
lexicographic rule, not a guarantee of finite-time unloading or reachability.
"""
import time
import numpy as np,scipy.sparse as sp,clarabel
from build.held_input_solver import violation
from build.contact_condensed_solver import solve as quadratic_solve


def solve(P,c,A,b,cones,nv,nu,nt,ns):
    began=time.perf_counter();load=np.zeros(len(c));load[nv+nu+nt+np.arange(0,ns,3)]=1.
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
    answer=clarabel.DefaultSolver(sp.csc_matrix(P.shape),load,sp.csc_matrix(A),b,cones,settings).solve()
    x=np.asarray(answer.x);error=violation(A,b,cones,x)
    if str(answer.status) not in ['Solved','AlmostSolved'] or not np.isfinite(error) or error>1e-5:
        raise ValueError('support-priority feasible solve: '+str(answer.status))
    minimum=max(0.,float(load@x));zero=minimum<=1e-8
    # Zero is an exact lower bound from the unilateral contact constraints.
    # Otherwise retain the computed optimum within the same force precision.
    limit=0. if zero else minimum+1e-8
    at=cones[0].dim+cones[1].dim
    matrix=np.insert(A,at,load,axis=0);bound=np.insert(b,at,limit)
    newcones=[cones[0],clarabel.NonnegativeConeT(cones[1].dim+1),*cones[2:]]
    result,note=quadratic_solve(P,c,matrix,bound,newcones,nv,nu,nt,fixed_zero_tail=ns if zero else 0)
    note.update(support_minimum_n=minimum,priority_wall_s=time.perf_counter()-began)
    return result,note
