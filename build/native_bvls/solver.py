"""Experimental full-rank BVLS binding, preserving objectives and box bounds."""
import ctypes
from pathlib import Path
import numpy as np
from scipy.optimize import OptimizeResult, lsq_linear as fallback
_lib=ctypes.CDLL(str(Path(__file__).with_name('bvls.so')));_call=_lib.bounded_solve
_p=ctypes.c_void_p;_i=ctypes.c_int
_call.argtypes=[_i,_i,_p,_p,_p,_p,ctypes.c_double,_i,_p,_p,_p,_p,_p];_call.restype=_i
calls=0;fallbacks=0

def solve(A,b,**options):
 global calls,fallbacks
 calls+=1
 if options.get('method')!='bvls' or set(options)-{'method','bounds','tol','max_iter'}:
  fallbacks+=1;return fallback(A,b,**options)
 A=np.ascontiguousarray(A,dtype=float);b=np.ascontiguousarray(b,dtype=float)
 m,n=A.shape;lo,hi=[np.ascontiguousarray(np.broadcast_to(x,(n,)),dtype=float)for x in options['bounds']]
 if m<n or n<1 or b.shape!=(m,) or not np.isfinite(A).all() or not np.isfinite(b).all() or np.any(lo>=hi) or np.isnan(lo).any() or np.isnan(hi).any():
  fallbacks+=1;return fallback(A,b,**options)
 x=np.empty(n);r=np.empty(m);active=np.empty(n);unbounded=np.empty(n);stats=np.empty(3)
 status=_call(m,n,A.ctypes.data,b.ctypes.data,lo.ctypes.data,hi.ctypes.data,float(options.get('tol',1e-10)),int(options.get('max_iter') or n),x.ctypes.data,r.ctypes.data,active.ctypes.data,unbounded.ctypes.data,stats.ctypes.data)
 if status<0 or not np.isfinite(x).all():
  fallbacks+=1;return fallback(A,b,**options)
 return OptimizeResult(x=x,fun=r,cost=stats[0],optimality=stats[1],active_mask=active,nit=int(stats[2]),status=status,success=status>0,message='native full-rank BVLS',unbounded_sol=(unbounded,None,n,None))
