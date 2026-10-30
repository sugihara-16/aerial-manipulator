"""Solve the same convex contact problem in independent whitened variables.

Norm cones are outer-approximated by supporting halfspaces. Accept only when
both the linear-QP optimality error and every original cone satisfy numerical
tolerances. Callers retain the original conic solver for unresolved problems.
"""
import time
from amsrr_det import native_qp as quadprog
import numpy as np, scipy.linalg as la, scipy.sparse as sp, clarabel
from build.held_input_solver import violation

def solve(P,c,A,b,cones,*args,mechanical_partition=None,workspace=None,mechanical_cache=None,**kwargs):
 began=time.perf_counter();n=len(c);ne=cones[0].dim
 E=A[:ne];row=np.maximum(np.linalg.norm(E,axis=1),1e-30)
 origin=basis=None
 if mechanical_partition is not None:
  nv,nu=mechanical_partition
  if ne>=nv:
   scaled=E/row[:,None];right=b[:ne]/row
   try:
    # Eliminate only velocity first. Rigid support reactions may be
    # redundant, so retain their null space with a smaller rank-revealing QR.
    core=mechanical_cache.get('core') if mechanical_cache is not None else None
    base=mechanical_cache.get('base',n) if mechanical_cache is not None else n
    if (core is not None and base<=n and np.array_equal(core,scaled[:nv,:base])
        and np.array_equal(mechanical_cache['right'],right[:nv]) and not np.any(scaled[:nv,base:])):
     response=np.pad(mechanical_cache['response'],((0,0),(0,n-base)))
    else:
     response=la.solve(scaled[:nv,:nv],np.c_[right[:nv],-scaled[:nv,nv:]],check_finite=False)
     if mechanical_cache is not None and base<=n and not np.any(scaled[:nv,base:]):
      mechanical_cache.update(core=scaled[:nv,:base].copy(),right=right[:nv].copy(),response=response[:,:1+base-nv].copy())
    shift,response=response[:,0],response[:,1:]
    remaining=scaled[nv:,nv:]+scaled[nv:,:nv]@response
    residual=right[nv:]-scaled[nv:,:nv]@shift
    if len(remaining):
     scale=np.maximum(np.linalg.norm(remaining,axis=1),1e-30)
     Q,R,order=la.qr((remaining/scale[:,None]).T,mode='full',pivoting=True,check_finite=False)
     diagonal=abs(np.diag(R));r=int(np.count_nonzero(diagonal>max(diagonal.max(initial=0.)*1e-12,1e-14)))
     rest=Q[:,:r]@la.solve_triangular(R[:r,:r].T,(residual/scale)[order[:r]],lower=True,check_finite=False)
     Z=Q[:,r:]
    else:rest=np.zeros(n-nv);Z=np.eye(n-nv);r=0
    origin=np.r_[shift+response@rest,rest];basis=np.vstack([response@Z,Z]);rank=nv+r
    if max(np.max(abs(E@origin-b[:ne]),initial=0.),np.max(abs(E@basis),initial=0.))>1e-7:
     origin=basis=None
   except la.LinAlgError:pass
 if basis is None:
  Q,R,order=la.qr((E/row[:,None]).T,mode='full',pivoting=True,check_finite=False)
  diagonal=abs(np.diag(R));rank=int(np.count_nonzero(diagonal>max(diagonal.max(initial=0.)*1e-12,1e-14)))
  origin=Q[:,:rank]@la.solve_triangular(R[:rank,:rank].T,(b[:ne]/row)[order[:rank]],lower=True,check_finite=False)
  basis=Q[:,rank:]
 H=basis.T@P@basis;linear=basis.T@(P@origin+c)
 if np.max(abs(E@origin-b[:ne]),initial=0.)>1e-7:raise ValueError('inconsistent contact equalities')
 factor=la.cholesky(H,lower=True,check_finite=False)
 transform=la.solve_triangular(factor,basis.T,lower=True,check_finite=False).T
 linear=la.solve_triangular(factor,linear,lower=True,check_finite=False);H=np.eye(len(linear))
 matrix=A[ne:]@transform;bound=b[ne:]-A[ne:]@origin;remaining=cones[1:];offset=0
 for cone in remaining:
  size=cone.dim;block=matrix[offset:offset+size]
  rows=np.maximum(np.maximum(np.linalg.norm(block,axis=1),abs(bound[offset:offset+size])),1e-12)
  if isinstance(cone,clarabel.SecondOrderConeT):rows[:]=rows.max()
  matrix[offset:offset+size]/=rows[:,None];bound[offset:offset+size]/=rows;offset+=size
 # Each supporting halfspace contains the exact norm cone. A feasible
 # optimum of this relaxation is also an optimum of the original cone QP.
 C=[];upper=[];norms=[];offset=0
 for cone in remaining:
  count=cone.dim;block=matrix[offset:offset+count];limit=bound[offset:offset+count];offset+=count
  if isinstance(cone,clarabel.NonnegativeConeT):C.append(block);upper.append(limit)
  elif isinstance(cone,clarabel.SecondOrderConeT):
   norms.append((block,limit));C.append(block[:1]);upper.append(limit[:1])
   for normal in np.r_[np.eye(count-1),-np.eye(count-1)]:
    C.append((block[0]-normal@block[1:])[None]);upper.append(np.array([limit[0]-normal@limit[1:]]))
  else:raise ValueError('unsupported reduced cone')
 for cut_round in range(16):
  constraints=C[0] if len(C)==1 else np.vstack(C) if C else np.empty((0,len(linear)))
  limits=upper[0] if len(upper)==1 else np.concatenate(upper) if upper else np.empty(0)
  solution=dual=None
  active=getattr(workspace,'contact_active_indices',None)
  if active is not None and np.all(active<len(limits)):
   W=constraints[active]
   try:
    multiplier=la.cho_solve(la.cho_factor(W@W.T,lower=True,check_finite=False),-limits[active]-W@linear,check_finite=False) if len(active) else np.empty(0)
    proposed=-linear-W.T@multiplier;full_dual=np.zeros(len(limits));full_dual[active]=multiplier
    slack=limits-constraints@proposed
    stationarity=proposed+linear+constraints.T@np.maximum(full_dual,0.)
    gap=.5*stationarity@stationarity+np.maximum(full_dual,0.)@np.maximum(slack,0.)
    if (np.min(slack,initial=0.)>=-1e-7 and gap<=1e-7
        and violation(A,b,cones,origin+transform@proposed)<=1e-7):
     solution,dual=proposed,full_dual;iterations=[0]
   except la.LinAlgError:pass
  if solution is None:
   solution,_,_,iterations,dual,_=quadprog.solve_qp(H,-linear,-constraints.T,-limits,factorized=True)
  dual=np.maximum(dual,0.)
  residual=max(float(np.max(constraints@solution-limits,initial=0.)),0.)
  stationarity=solution+linear+constraints.T@dual
  gap=.5*stationarity@stationarity+dual@np.maximum(limits-constraints@solution,0.)
  if residual>1e-7 or gap>1e-7:raise ValueError('cut QP residual/gap '+str((residual,gap)))
  violations=[]
  for block,limit in norms:
   slack=limit-block@solution;norm=np.linalg.norm(slack[1:])
   if norm-slack[0]>1e-12:
    direction=slack[1:]/norm;violations.append((block[0]-direction@block[1:],limit[0]-direction@limit[1:]))
  if not violations:break
  for row,value in violations:C.append(row[None]);upper.append(np.array([value]))
 else:raise ValueError('norm cuts unresolved')
 z=origin+transform@solution;error=violation(A,b,cones,z)
 if not np.isfinite(error)or error>1e-7:raise ValueError('QR contact solve residual:'+str(error))
 if workspace is not None:
  workspace.contact_active_indices=np.flatnonzero(dual>0.)
 return z,dict(wall_s=time.perf_counter()-began,iterations=int(iterations[0]),cut_rounds=cut_round,objective_gap=float(gap),original_residual=error,variables=basis.shape[1],equality_rank=rank)
