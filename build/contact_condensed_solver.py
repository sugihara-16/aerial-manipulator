"""Exact tangent-reaction elimination and diagonal scaling of a contact QP.

Objective and every force/input cone are unchanged. Reconstruction is checked
against the original constraints; failed numerical solves use the original QP.
"""
import time
import numpy as np
import scipy.sparse as sp
import clarabel
from build.held_input_solver import violation

STATISTICS=dict(calls=0,fallbacks=0,statuses={},largest_original_residual=0.)


def scaled_recovery(P,c,A,b,cones):
    """Same optimization with bounded row scaling; never amplify empty rows."""
    rows=np.maximum(1.,np.maximum(np.max(abs(A),axis=1),abs(b)));offset=0
    for cone in cones:
        # A norm cone must be scaled as a whole to preserve its geometry.
        if isinstance(cone,clarabel.SecondOrderConeT):rows[offset:offset+cone.dim]=max(rows[offset:offset+cone.dim])
        offset+=cone.dim
    cost=max(1.,float(abs(P).max()),float(abs(c).max()))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.tol_feas=1e-11;settings.tol_gap_abs=1e-14;settings.tol_gap_rel=1e-10
    settings.reduced_tol_feas=1e-9;settings.reduced_tol_gap_abs=1e-13;settings.reduced_tol_gap_rel=1e-8
    solver=clarabel.DefaultSolver(sp.csc_matrix(np.triu(P/cost)),c/cost,
        sp.csc_matrix(A/rows[:,None]),b/rows,cones,settings)
    answer=solver.solve();x=np.asarray(answer.x);residual=violation(A,b,cones,x)
    if str(answer.status)not in ['Solved','AlmostSolved']or not np.isfinite(residual)or residual>1e-7:
        raise ValueError('contact solve '+str(answer.status))
    return answer,dict(original_residual=residual,original_objective_gap_reported=float(solver.get_info().gap_abs*cost))


def solve(P,c,A,b,cones,nv,nu,nt,*,fixed_zero_tail=0):
    if fixed_zero_tail:
        # A zero support-normal sum plus nonnegative pressure and friction
        # implies every support reaction is exactly zero. Remove this face
        # of the cone rather than asking an interior-point solve to find it.
        cut=len(c)-fixed_zero_tail;reduced=A[:,:cut]
        selected=[];offset=0;reduced_cones=[]
        for cone in cones:
            ids=np.arange(offset,offset+cone.dim);offset+=cone.dim
            kind=type(cone).__name__
            if kind in ['ZeroConeT','NonnegativeConeT']:
                zero=np.all(reduced[ids]==0.,axis=1)
                valid=(b[ids][zero]==0.) if kind=='ZeroConeT' else (b[ids][zero]>=0.)
                if not np.all(valid):raise ValueError('inconsistent zero-support constraint')
                ids=ids[~zero]
            selected.extend(ids);reduced_cones.append(type(cone)(len(ids)))
        x,note=solve(P[:cut,:cut],c[:cut],reduced[selected],b[selected],reduced_cones,nv,nu,nt)
        x=np.r_[x,np.zeros(fixed_zero_tail)]
        residual=violation(A,b,cones,x)
        if not np.isfinite(residual) or residual>1e-5:raise ValueError('zero-support reconstruction residual')
        note.update(fixed_zero_support_variables=fixed_zero_tail,original_residual=residual)
        return x,note
    begin=time.perf_counter();size=len(c)
    columns=np.arange(nv+nu,nv+nu+nt);rows=np.arange(nv,nv+nt)
    keep=np.r_[np.arange(nv+nu),np.arange(nv+nu+nt,size)]
    retained_rows=np.r_[np.arange(nv),np.arange(nv+nt,len(b))]
    transform=np.eye(size)[:,keep];origin=np.zeros(size)
    if nt:
        diagonal=A[rows,columns]
        if np.any(diagonal==0) or np.count_nonzero(A[np.ix_(rows,columns)])!=nt:
            raise ValueError('contact condensation requires diagonal tangent compliance')
        origin[columns]=b[rows]/diagonal
        transform[columns]=-A[np.ix_(rows,keep)]/diagonal[:,None]
    H=transform.T@P@transform;f=transform.T@(P@origin+c)
    scale=1/np.sqrt(H.diagonal());transform*=scale
    H*=scale[:,None]*scale[None,:];f*=scale
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
    reduced_cones=[clarabel.ZeroConeT(cones[0].dim-nt),*cones[1:]]
    answer=clarabel.DefaultSolver(sp.csc_matrix(np.triu(H)),f,
        sp.csc_matrix(A[retained_rows]@transform),b[retained_rows]-A[retained_rows]@origin,
        reduced_cones,settings).solve()
    x=origin+transform@np.asarray(answer.x)
    residual=violation(A,b,cones,x)
    STATISTICS['calls']+=1
    status=str(answer.status);STATISTICS['statuses'][status]=STATISTICS['statuses'].get(status,0)+1
    STATISTICS['largest_original_residual']=max(STATISTICS['largest_original_residual'],residual)
    # Match the original controller's status/residual contract. It separately
    # rechecks both physical pad norm limits after this reconstruction.
    fallback=(status not in ['Solved','AlmostSolved'] or not np.isfinite(residual) or residual>1e-5)
    recovery=None
    if fallback:
        STATISTICS['fallbacks']+=1
        answer=clarabel.DefaultSolver(sp.csc_matrix(np.triu(P)),c,sp.csc_matrix(A),b,cones,settings).solve()
        x=np.asarray(answer.x);residual=violation(A,b,cones,x)
        if str(answer.status)not in ['Solved','AlmostSolved']or not np.isfinite(residual)or residual>1e-5:
            answer,recovery=scaled_recovery(P,c,A,b,cones)
            x=np.asarray(answer.x);residual=violation(A,b,cones,x)
    return x,dict(wall_s=time.perf_counter()-begin,variables=len(keep),fallback=fallback,
                  iterations=int(answer.iterations),original_residual=residual,scaled_recovery=recovery)
