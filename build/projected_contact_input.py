"""Nearest rated input subject to the original finite one-period conditions.

Velocity and finite reactions are uniquely reconstructed for every drive
input. No response rollout or evaluator force is used. All original rows and
pad norm cones are checked after the numerical solve.
"""
import time
import numpy as np,scipy.sparse as sp,clarabel
from scipy.linalg import solve
from build.held_input_solver import violation


def project(matrix,bound,cones,nv,nu,target,*,motion_indices=None,motion_goal=None,input_weight=1e-6):
    began=time.perf_counter();size=matrix.shape[1]
    inputs=np.arange(nv,nv+nu);unknown=np.r_[np.arange(nv),np.arange(nv+nu,size)]
    ne=cones[0].dim
    if ne!=len(unknown):raise ValueError('projection requires the full finite contact response')
    equations=matrix[:ne,unknown];rhs=np.c_[bound[:ne],-matrix[:ne,inputs]]
    row_scale=np.maximum(np.linalg.norm(equations,axis=1),1e-30)
    response=solve(equations/row_scale[:,None],rhs/row_scale[:,None],check_finite=True)
    origin=np.zeros(size);origin[unknown]=response[:,0]
    mapping=np.zeros((size,nu));mapping[inputs]=np.eye(nu);mapping[unknown]=response[:,1:]
    H=np.eye(nu);linear=-target
    if motion_indices is not None:
        output=mapping[motion_indices];offset=origin[motion_indices]-motion_goal
        H=output.T@output+input_weight*np.eye(nu)
        linear=output.T@offset-input_weight*target
    optimum=target if motion_indices is None else np.linalg.solve(H,-linear)
    candidate=origin+mapping@optimum
    residual=violation(matrix,bound,cones,candidate)
    status='unmodified_input' if motion_indices is None else 'unconstrained_optimum';iterations=0
    if not np.isfinite(residual)or residual>1e-5:
        reduced=matrix[ne:]@mapping;limits=bound[ne:]-matrix[ne:]@origin
        # Equivalent row scaling, with one common scale per norm cone.
        start=0
        for cone in cones[1:]:
            n=cone.dim;block=reduced[start:start+n]
            scale=np.maximum(np.maximum(np.linalg.norm(block,axis=1),abs(limits[start:start+n])),1e-12)
            if isinstance(cone,clarabel.SecondOrderConeT):scale[:]=scale.max()
            reduced[start:start+n]/=scale[:,None];limits[start:start+n]/=scale;start+=n
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
        answer=clarabel.DefaultSolver(sp.csc_matrix(np.triu(H)),linear,sp.csc_matrix(reduced),limits,cones[1:],settings).solve()
        status=str(answer.status);iterations=int(answer.iterations)
        if status not in ['Solved','AlmostSolved']:raise ValueError('projected contact input '+status)
        candidate=origin+mapping@np.asarray(answer.x)
        residual=violation(matrix,bound,cones,candidate)
    if not np.isfinite(residual)or residual>1e-5:raise ValueError('projected contact input original residual '+str(residual))
    return candidate,dict(wall_s=time.perf_counter()-began,status=status,iterations=iterations,
        original_residual=float(residual),normalized_input_correction=float(np.linalg.norm(candidate[inputs]-target)),
        normalized_motion_error=None if motion_indices is None else float(np.linalg.norm(candidate[motion_indices]-motion_goal)))
