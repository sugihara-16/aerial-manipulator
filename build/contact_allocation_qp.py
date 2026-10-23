"""Solve the same convex input allocation after eliminating its equalities.

Diagonal variable and constraint scaling changes no objective or constraint.
Check the original units and a convex objective-error bound after solving.
"""
import numpy as np
import scipy.sparse as sp
import clarabel


def solve(H,g,E,rhs,C,low,high):
    Q,R=np.linalg.qr(E.T,mode='complete');rank=len(rhs)
    origin=Q[:,:rank]@np.linalg.solve(R[:rank].T,rhs);Z=Q[:,rank:]
    P=Z.T@H@Z;linear=Z.T@(H@origin+g)
    scale=1/np.sqrt(np.maximum(np.diag(P),max(np.max(np.diag(P)),1e-30)*1e-12))
    transform=Z*scale
    matrix=np.r_[C@transform,-C@transform];bound=np.r_[high-C@origin,C@origin-low]
    nonzero=np.any(matrix!=0.,axis=1)
    if np.any(bound[~nonzero]<-1e-7):raise ValueError("constant constraint infeasible")
    row_scale=np.maximum(np.linalg.norm(matrix,axis=1),1e-12)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.static_regularization_constant=1e-12
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-11
    answer=clarabel.DefaultSolver(sp.csc_matrix(np.triu(P*scale[:,None]*scale)),linear*scale,
        sp.csc_matrix(matrix[nonzero]/row_scale[nonzero,None]),bound[nonzero]/row_scale[nonzero],
        [clarabel.NonnegativeConeT(int(nonzero.sum()))],settings).solve()
    if str(answer.status)not in ('Solved','AlmostSolved'):
        raise ValueError('contact allocation '+str(answer.status))
    x=origin+transform@np.asarray(answer.x)
    dual=np.zeros(len(bound));dual[nonzero]=np.maximum(np.asarray(answer.z)/row_scale[nonzero],0.)
    y=np.linalg.lstsq(E.T,-(H@x+g+C.T@(dual[:len(low)]-dual[len(low):])),rcond=1e-12)[0]
    residual=float(max(np.max(abs(E@x-rhs),initial=0.),np.max(low-C@x),np.max(C@x-high)))
    if not np.isfinite(residual) or residual>1e-7:
        raise ValueError('contact allocation residual '+str(residual))
    # C starts with identity actuator bounds. Minimize the Lagrangian's affine
    # tangent over that box to bound f(x)-f(optimum), including pad multipliers.
    stationarity=H@x+g+C.T@(dual[:len(low)]-dual[len(low):])+E.T@y
    n=len(x)
    gap=float(np.maximum(stationarity,0.)@np.maximum(x-low[:n],0.)+
              np.maximum(-stationarity,0.)@np.maximum(high[:n]-x,0.)+
              dual@np.maximum(np.r_[high-C@x,C@x-low],0.)+abs(y@(E@x-rhs)))
    gap/=max(float(np.diag(H).max()),1.)
    if not np.isfinite(gap) or gap>1e-8:
        raise ValueError('contact allocation objective unresolved: '+str(gap))
    return x,residual,gap
