"""Keep small HiGHS solves single-threaded inside parallel structure trials."""
import warnings
from scipy.optimize import linprog as _linprog, OptimizeWarning


def linprog(*args,**kwargs):
    kwargs['options']=dict(kwargs.get('options') or {},threads=1)
    # SciPy passes this supported HiGHS option through, with a wrapper warning.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message=r"Unrecognized options detected: .*'threads': 1.*",category=OptimizeWarning)
        return _linprog(*args,**kwargs)


def conic_linprog(c,*,A_ub,b_ub,A_eq,b_eq,bounds,**kwargs):
    """Small dense LP using the existing conic solver, with checked fallback.

    An unsuccessful conic solve is not used as an infeasibility certificate.
    Callers requiring HiGHS sensitivities must use linprog instead.
    """
    import numpy as np
    import scipy.sparse as sp
    import clarabel
    from types import SimpleNamespace
    n=len(c);unit=np.eye(n)
    lo=np.array([-np.inf if x is None else x for x,y in bounds])
    hi=np.array([np.inf if y is None else y for x,y in bounds])
    il=np.isfinite(lo);ih=np.isfinite(hi)
    A=np.vstack([A_eq,A_ub,unit[ih],-unit[il]])
    b=np.r_[b_eq,b_ub,hi[ih],-lo[il]]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
    settings.max_iter=100
    if 'time_limit' in (kwargs.get('options') or {}):settings.time_limit=kwargs['options']['time_limit']
    solved=clarabel.DefaultSolver(sp.csc_matrix((n,n)),np.asarray(c),sp.csc_matrix(A),b,
        [clarabel.ZeroConeT(len(A_eq)),clarabel.NonnegativeConeT(len(A)-len(A_eq))],settings).solve()
    # Return bounded commands; recheck equality and inequality residuals
    # after removing tiny solver roundoff at a bound.
    x=np.clip(np.asarray(solved.x),lo,hi)
    error=max(float(abs(A_eq@x-b_eq).max(initial=0.)),float((A_ub@x-b_ub).max(initial=0.)),
              float((lo-x).max(initial=0.)),float((x-hi).max(initial=0.)))
    if str(solved.status) in ['Solved','AlmostSolved'] and error<1e-8:
        return SimpleNamespace(success=True,status=0,message='checked conic LP',nit=solved.iterations,x=x,fun=float(np.dot(c,x)))
    return linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,**kwargs)
