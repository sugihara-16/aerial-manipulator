"""Solve the same convex contact problem in independent whitened variables.

Norm cones are outer-approximated by supporting halfspaces. Accept only when
both the linear-QP optimality error and every original cone satisfy numerical
tolerances. Callers retain the original conic solver for unresolved problems.
"""
import time,quadprog
import numpy as np, scipy.linalg as la, scipy.sparse as sp, clarabel
from build.held_input_solver import violation

def solve(P,c,A,b,cones,*args,**kwargs):
 began=time.perf_counter();n=len(c);ne=cones[0].dim
 E=A[:ne];row=np.maximum(np.linalg.norm(E,axis=1),1e-30)
 Q,R,order=la.qr((E/row[:,None]).T,mode='full',pivoting=True,check_finite=False)
 diagonal=abs(np.diag(R));rank=int(np.count_nonzero(diagonal>max(diagonal.max(initial=0.)*1e-12,1e-14)))
 origin=Q[:,:rank]@la.solve_triangular(R[:rank,:rank].T,(b[:ne]/row)[order[:rank]],lower=True,check_finite=False)
 basis=Q[:,rank:];H=basis.T@P@basis;linear=basis.T@(P@origin+c)
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
  if isinstance(cone,clarabel.NonnegativeConeT):C.extend(block);upper.extend(limit)
  elif isinstance(cone,clarabel.SecondOrderConeT):
   norms.append((block,limit));C.append(block[0]);upper.append(limit[0])
   for normal in np.r_[np.eye(count-1),-np.eye(count-1)]:
    C.append(block[0]-normal@block[1:]);upper.append(limit[0]-normal@limit[1:])
  else:raise ValueError('unsupported reduced cone')
 for cut_round in range(16):
  constraints=np.asarray(C);limits=np.asarray(upper)
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
  for row,value in violations:C.append(row);upper.append(value)
 else:raise ValueError('norm cuts unresolved')
 z=origin+transform@solution;error=violation(A,b,cones,z)
 if not np.isfinite(error)or error>1e-7:raise ValueError('QR contact solve residual:'+str(error))
 return z,dict(wall_s=time.perf_counter()-began,iterations=int(iterations[0]),cut_rounds=cut_round,objective_gap=float(gap),original_residual=error,variables=basis.shape[1],equality_rank=rank)
