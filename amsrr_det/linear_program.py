"""Keep small HiGHS solves single-threaded inside parallel structure trials."""
import warnings
from scipy.optimize import linprog as _linprog, OptimizeWarning


def linprog(*args,**kwargs):
    kwargs['options']=dict(kwargs.get('options') or {},threads=1)
    # SciPy passes this supported HiGHS option through, with a wrapper warning.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message=r"Unrecognized options detected: .*'threads': 1.*",category=OptimizeWarning)
        return _linprog(*args,**kwargs)


def conic_linprog(c,*,A_ub,b_ub,A_eq,b_eq,bounds,reusable=None,**kwargs):
    """Small dense LP using the existing conic solver, with checked fallback.

    An unsuccessful conic solve is not used as an infeasibility certificate.
    Callers requiring HiGHS sensitivities must use linprog instead.
    """
    import numpy as np
    import scipy.sparse as sp
    import clarabel
    from types import SimpleNamespace
    n=len(c)
    lo=np.array([-np.inf if x is None else x for x,y in bounds])
    hi=np.array([np.inf if y is None else y for x,y in bounds])
    il=np.isfinite(lo);ih=np.isfinite(hi)
    b=np.r_[b_eq,b_ub,hi[ih],-lo[il]]
    limit=(kwargs.get('options') or {}).get('time_limit')
    key=None if reusable is None else ('lp',np.asarray(c).tobytes(),A_eq.shape,A_eq.tobytes(),
        A_ub.shape,A_ub.tobytes(),lo.tobytes(),hi.tobytes(),limit)
    solver=None if reusable is None else reusable.get(key)
    if solver is None:
        unit=np.eye(n);A=np.vstack([A_eq,A_ub,unit[ih],-unit[il]])
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
        settings.max_iter=100
        if limit is not None:settings.time_limit=limit
        solver=clarabel.DefaultSolver(sp.csc_matrix((n,n)),np.asarray(c),sp.csc_matrix(A),b,
            [clarabel.ZeroConeT(len(A_eq)),clarabel.NonnegativeConeT(len(A)-len(A_eq))],settings)
        if reusable is not None and solver.is_data_update_allowed():reusable[key]=solver
    else:
        solver.update(b=b)
    solved=solver.solve()
    # Return bounded commands; recheck equality and inequality residuals
    # after removing tiny solver roundoff at a bound.
    x=np.clip(np.asarray(solved.x),lo,hi)
    error=max(float(abs(A_eq@x-b_eq).max(initial=0.)),float((A_ub@x-b_ub).max(initial=0.)),
              float((lo-x).max(initial=0.)),float((x-hi).max(initial=0.)))
    if str(solved.status) in ['Solved','AlmostSolved'] and error<1e-8:
        return SimpleNamespace(success=True,status=0,message='checked conic LP',nit=solved.iterations,x=x,fun=float(np.dot(c,x)))
    return linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,**kwargs)


def reused_simplex_linprog(c,*,A_ub,b_ub,A_eq,b_eq,bounds,reusable=None,**kwargs):
    """Reuse a simplex basis when only the required load changes.

    Every returned solution is checked against the original matrices and
    bounds. Nonoptimal or inaccurate results fall back to the existing solve.
    """
    import numpy as np
    import highspy
    from scipy.sparse import csc_matrix
    from types import SimpleNamespace
    lo=np.array([-np.inf if a is None else a for a,b in bounds])
    hi=np.array([np.inf if b is None else b for a,b in bounds])
    lower=np.r_[b_eq,np.full(len(b_ub),-np.inf)];upper=np.r_[b_eq,b_ub]
    options=kwargs.get('options') or {}
    limit=options.get('time_limit',float('inf'));iterations=options.get('maxiter',2147483647)
    key=('simplex',np.asarray(c).tobytes(),A_eq.shape,A_eq.tobytes(),A_ub.shape,A_ub.tobytes(),lo.tobytes(),hi.tobytes(),limit,iterations)
    solver=None if reusable is None else reusable.get(key)
    if solver is None:
        solver=highspy.Highs()
        for name,value in [('output_flag',False),('threads',1),('solver','simplex'),
                           ('primal_feasibility_tolerance',1e-9),('dual_feasibility_tolerance',1e-9),
                           ('time_limit',limit),('simplex_iteration_limit',iterations)]:
            solver.setOptionValue(name,value)
        A=csc_matrix(np.vstack([A_eq,A_ub]));model=highspy.HighsLp()
        model.num_col_=len(c);model.num_row_=len(lower);model.col_cost_=c
        model.col_lower_=lo;model.col_upper_=hi;model.row_lower_=lower;model.row_upper_=upper
        model.a_matrix_.format_=highspy.MatrixFormat.kColwise
        model.a_matrix_.start_=A.indptr;model.a_matrix_.index_=A.indices;model.a_matrix_.value_=A.data
        solver.passModel(model)
        if reusable is not None:reusable[key]=solver
    else:
        solver.changeRowsBounds(len(lower),np.arange(len(lower),dtype=np.int32),lower,upper)
    solver.run();x=np.clip(np.array(solver.getSolution().col_value),lo,hi)
    error=max(abs(A_eq@x-b_eq).max(initial=0.),(A_ub@x-b_ub).max(initial=0.))
    if solver.getModelStatus()!=highspy.HighsModelStatus.kOptimal or not np.isfinite(x).all() or error>=1e-8:
        return conic_linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,**kwargs)
    return SimpleNamespace(success=True,status=0,message='checked reused simplex',
                           nit=solver.getInfo().simplex_iteration_count,x=x,fun=float(c@x))
