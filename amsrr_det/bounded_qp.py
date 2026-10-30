"""One bounded convex allocation, with feasible output on numerical expiry."""
import ctypes
import time
import numpy as np
from amsrr_det.native_qp import _library as lib, _solve as call
lib.qp_set_deadline.argtypes=[ctypes.c_double]

def bounded_qp(H,a,C,b,lower,upper,previous,*,deadline,friction=2.25):
 n=len(a);nu=len(lower);count=len(b);r=min(n,count)
 def project(value):
  x=np.nan_to_num(value.copy(),nan=0.,posinf=0.,neginf=0.);x[:nu]=np.clip(x[:nu],lower,upper)
  if n>nu:
   f=x[nu:].reshape(-1,3);f[:,0]=np.maximum(0,f[:,0]);norm=np.linalg.norm(f[:,1:],axis=1);scale=np.minimum(1.,friction*np.cos(np.pi/8)*f[:,0]/np.maximum(norm,1e-20));f[:,1:]*=scale[:,None]
  return x
 incumbent=project(previous)
 if time.perf_counter()>=deadline:return incumbent,True,0
 G=np.array(H,order='F',dtype=float);a_work=np.array(a,dtype=float);matrix=np.array(C,order='F',dtype=float);bound=np.array(b,dtype=float)
 x=incumbent.copy();dual=np.zeros(count);cost=np.zeros(1);active=np.zeros(count,np.int32);nact=np.zeros(1,np.int32);iterations=np.zeros(2,np.int32);work=np.empty(2*n+2*count+r*(r+5)//2)
 lib.qp_set_deadline(float(deadline))
 status=call(G.ctypes.data,a_work.ctypes.data,n,x.ctypes.data,dual.ctypes.data,cost.ctypes.data,matrix.ctypes.data,bound.ctypes.data,count,0,active.ctypes.data,nact.ctypes.data,iterations.ctypes.data,work.ctypes.data,0)
 lib.qp_set_deadline(0.)
 if status==0:return x,False,int(iterations[0])
 if status==3:
  candidate=project(x)
  if .5*candidate@H@candidate-a@candidate < .5*incumbent@H@incumbent-a@incumbent:incumbent=candidate
 return incumbent,True,int(iterations[0])
