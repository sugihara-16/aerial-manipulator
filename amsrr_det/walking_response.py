"""Frozen whole-body response of the actual zero-order-held joint servos.

The command is constant for a control period. Joint torques therefore change
as the measured configuration/velocity leave the observation at which the
command was issued. Full inertia coupling is retained. Contact forces are
allocation variables, not simulator measurements. Saturation/geometry changes
within the interval are outside this local response model.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .bounded_qp import bounded_qp

def response(mass,vi,kp,kd,velocity,*,period=.01,step=.0005):
 n=len(mass);count=max(1,round(period/step));h=period/count
 p=np.zeros(n);d=np.zeros(n);p[vi]=kp;d[vi]=kd
 factor=cho_factor(mass+np.diag(h*d),lower=True,check_finite=False)
 inv=cho_solve(factor,np.eye(n),check_finite=False)
 # Same implicit damping / explicit position feedback as implicitfast.
 A=inv@mass;C=-h*inv*p[None,:];F=h*inv
 vf=np.zeros((n,n));qf=vf.copy();vb=np.zeros(n);qb=vb.copy()
 for _ in range(count):
  vf=A@vf+C@qf+F;vb=A@vb+C@qb
  qf+=h*vf;qb+=h*(velocity+vb)
 return vf/period,vb/period,qf,qb


def minimum_support_qp(H,a,C,b,lower,upper,previous,*,support_row,support_target,**kwargs):
 if support_row is None:return bounded_qp(H,a,C,b,lower,upper,previous,**kwargs)
 n=len(a);nu=len(lower);indices=np.r_[np.arange(nu),np.arange(nu+1,n+1)];M=np.zeros((n+1,n+1));M[np.ix_(indices,indices)]=H;M[nu,indices]=M[indices,nu]=-100*support_row;M[nu,nu]=10000.+1e-8
 v=np.zeros(n+1);v[indices]=a;v[nu]=-100*support_target
 constraints=np.zeros((n+1,C.shape[1]+2));constraints[indices,:-2]=C;constraints[nu,-2:]=[1.,-1.]
 start=np.r_[previous[:nu],max(0.,(support_row@previous-support_target)/100),previous[nu:]]
 x,fall,it=bounded_qp(M,v,constraints,np.r_[b,0.,-1e6],np.r_[lower,0.],np.r_[upper,1e6],start,**kwargs)
 return x[indices],fall,it
