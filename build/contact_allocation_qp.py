"""Solve one convex input allocation in equality-preserving coordinates.

Scaling and numerical fallbacks preserve the physical constraints. A candidate
must pass original-unit feasibility and a convex objective-error bound.
"""
import numpy as np
import scipy.sparse as sp
import clarabel
import quadprog


def solve(H,g,E,rhs,C,low,high):
    Q,R=np.linalg.qr(E.T,mode='complete');rank=len(rhs)
    origin=Q[:,:rank]@np.linalg.solve(R[:rank].T,rhs);Z=Q[:,rank:]
    P=Z.T@H@Z;linear=Z.T@(H@origin+g)
    scale=1/np.sqrt(np.maximum(np.diag(P),max(np.max(np.diag(P)),1e-30)*1e-12))
    transform=Z*scale;P=P*scale[:,None]*scale;linear=linear*scale
    matrix=np.r_[C@transform,-C@transform];bound=np.r_[high-C@origin,C@origin-low]
    nonzero=np.any(matrix!=0.,axis=1)
    if np.any(bound[~nonzero]<-1e-7):raise ValueError('constant constraint infeasible')
    row_scale=np.maximum(np.linalg.norm(matrix,axis=1),1e-12)
    A=matrix[nonzero]/row_scale[nonzero,None];b=bound[nonzero]/row_scale[nonzero]

    def checked(z,multipliers):
        x=origin+transform@z
        dual=np.zeros(len(bound));dual[nonzero]=np.maximum(multipliers/row_scale[nonzero],0.)
        y=np.linalg.lstsq(E.T,-(H@x+g+C.T@(dual[:len(low)]-dual[len(low):])),rcond=1e-12)[0]
        residual=float(max(np.max(abs(E@x-rhs),initial=0.),np.max(low-C@x),np.max(C@x-high)))
        if not np.isfinite(residual) or residual>1e-7:
            raise ValueError('contact allocation residual '+str(residual))
        # C starts with identity actuator bounds. Minimize the Lagrangian's
        # affine tangent over that box to bound the ORIGINAL objective error.
        stationarity=H@x+g+C.T@(dual[:len(low)]-dual[len(low):])+E.T@y;n=len(x)
        gap=float(np.maximum(stationarity,0.)@np.maximum(x-low[:n],0.)+
                  np.maximum(-stationarity,0.)@np.maximum(high[:n]-x,0.)+
                  dual@np.maximum(np.r_[high-C@x,C@x-low],0.)+abs(y@(E@x-rhs)))
        gap/=max(float(np.diag(H).max()),1.)
        if not np.isfinite(gap) or gap>1e-8:
            raise ValueError('contact allocation objective unresolved: '+str(gap))
        return x,residual,gap

    # The active-set solver needs positive curvature. Tiny diagonal terms are
    # numerical aids only: checked() still uses the unmodified H and g, so no
    # new objective or relaxed constraint is accepted through regularization.
    for ridge in (1e-12,1e-10,1e-8):
        try:
            z,_,_,_,dual,_=quadprog.solve_qp(P+ridge*np.eye(len(P)),-linear,-A.T,-b)
            return checked(z,dual)
        except (ValueError,np.linalg.LinAlgError):
            pass
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.static_regularization_constant=1e-12
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-11
    answer=clarabel.DefaultSolver(sp.csc_matrix(np.triu(P)),linear,sp.csc_matrix(A),b,
        [clarabel.NonnegativeConeT(int(nonzero.sum()))],settings).solve()
    error=ValueError('contact allocation '+str(answer.status))
    if str(answer.status) in ('Solved','AlmostSolved'):
        try:return checked(np.asarray(answer.x),np.asarray(answer.z))
        except ValueError as failure:error=failure
    raise error
