"""Periodic whole-body path smoothing and immutable SE(2) path evaluation."""
import numpy as np,mujoco
from .sparse_least_squares import least_squares
from scipy.sparse import bsr_matrix,csr_matrix,vstack,kron,lil_matrix
from .walking_native import CycleGeometry
from .walking import rotation_z
from .walking_gait import foot_targets,foot_headings


class CycleReference:
 """Repeat internal motion while following the requested world path.

 The stored arrays remain unchanged. Removing world travel twice introduces
 an artificial velocity jump at each cycle boundary, so normalization lives
 only here, separate from the physical configuration samples.
 """
 def __init__(self,plant,spec,gait):
  from scipy.interpolate import CubicSpline
  self.m=plant.mj;self.spec=spec;self.gait=gait;self.period=float(gait['period'])
  self.origin=np.asarray(gait['origin']);values=np.array(gait['configurations'],copy=True)
  times=np.linspace(0,self.period,len(values));rot=np.empty((3,3))
  for time,q in zip(times,values):
   xy,yaw,*_=spec.path(time);R=rotation_z(-yaw)
   q[:3]=self.origin+R@(q[:3]-self.origin-np.r_[xy,0.])
   mujoco.mju_quat2Mat(rot.ravel(),q[3:7]);mujoco.mju_mat2Quat(q[3:7],(R@rot).ravel())
  for k in range(1,len(values)):
   if values[k,3:7]@values[k-1,3:7]<0:values[k,3:7]*=-1
  values[-1]=values[0]
  self.spline=CubicSpline(times,values,bc_type='periodic')

 def pose(self,time):
  q=self.spline(time%self.period);xy,yaw,*_=self.spec.path(time);R=rotation_z(yaw)
  q[:3]=self.origin+R@(q[:3]-self.origin)+np.r_[xy,0.]
  mujoco.mj_normalizeQuat(self.m,q);rot=np.empty((3,3))
  mujoco.mju_quat2Mat(rot.ravel(),q[3:7]);mujoco.mju_mat2Quat(q[3:7],(R@rot).ravel())
  return q

 def rates(self,time):
  h=1e-4;q=self.pose(time);before=self.pose(time-h);after=self.pose(time+h)
  v=np.empty(self.m.nv);a=np.empty(self.m.nv);b=np.empty(self.m.nv)
  mujoco.mj_differentiatePos(self.m,v,2*h,before,after)
  mujoco.mj_differentiatePos(self.m,a,h,before,q);mujoco.mj_differentiatePos(self.m,b,h,q,after)
  return q,v,(b-a)/h

 def feet(self,time):
  return foot_targets(time,self.period,self.spec.speed,self.gait['feet'],height=self.gait['height'],
                      curvature=self.spec.curvature,origin=self.origin,duty=self.gait.get('duty',.62),phases=self.gait.get('phases'))

def smooth_cycle(kin,spec,gait,weight=.008):
 m,d=kin.m,kin.d;idx=kin.indices;nd=len(idx);count=len(gait['configurations'])-1;dt=gait['period']/count
 bases=[];targets=[];rotations=[];supports=[];x0=[]
 for k in range(count):
  t=k*dt;xy,yaw,*_=spec.path(t);base=kin.nominal.copy()
  base[:3]=gait['origin']+rotation_z(yaw)@(base[:3]-gait['origin'])+np.r_[xy,0.]
  R=np.empty((3,3));mujoco.mju_quat2Mat(R.ravel(),base[3:7]);mujoco.mju_mat2Quat(base[3:7],(rotation_z(yaw)@R).ravel())
  bases.append(base);delta=np.zeros(m.nv);mujoco.mj_differentiatePos(m,delta,1,base,gait['configurations'][k]);x0.append(delta[idx])
  feet=foot_targets(t,gait['period'],spec.speed,gait['feet'],height=gait['height'],curvature=spec.curvature,origin=gait['origin'],duty=gait.get('duty',.62),phases=gait.get('phases'))
  targets.append(feet[0]);supports.append(feet[3])
  rotations.append([rotation_z(a)@R for a,R in zip(foot_headings(t,gait['period'],spec.speed,len(gait['feet']),spec.curvature,duty=gait.get('duty',.62),phases=gait.get('phases')),kin.rotations)])
 x0=np.array(x0);D=lil_matrix((count,count))
 for k in range(count):D[k,k]=-2;D[k,(k-1)%count]=1;D[k,(k+1)%count]=1
 weights=np.r_[np.full(3,weight*4),np.full(3,weight*2),np.full(nd-6,weight)]
 smoothmap=kron(D.tocsr()/dt**2,csr_matrix(np.diag(weights)))
 lo=np.full((count,nd),-np.inf);hi=-lo
 lo[:,6:]=kin.limits[:,0]-kin.nominal[kin.qi];hi[:,6:]=kin.limits[:,1]-kin.nominal[kin.qi]
 com_targets=np.array([gait['origin'][:2]+spec.path(k*dt)[0] for k in range(count)])
 geometry=CycleGeometry(kin)
 last=None;pair=None
 def evaluate(flat):
  nonlocal last,pair
  if last is not None and np.array_equal(last,flat):return pair
  X=flat.reshape(count,nd);res=[];blocks=[]
  for k,x in enumerate(X):
   residual,jacobian=geometry.evaluate(x,bases[k],targets[k],rotations[k],supports[k],com_targets[k])
   for i,active in enumerate(supports[k]):
    if not active:
     residual[12*i:12*i+2]=0.;jacobian[12*i:12*i+2]=0.
     if residual[12*i+2]>0:residual[12*i+2]=0.;jacobian[12*i+2]=0.
   res.append(residual);blocks.append(jacobian)
  res=np.r_[np.concatenate(res),smoothmap@flat];matrix=bsr_matrix((np.array(blocks),np.arange(count),np.arange(count+1)),shape=(count*blocks[0].shape[0],count*nd)).tocsr();matrix.eliminate_zeros();J=vstack([matrix,smoothmap]).tocsr()
  last=flat.copy();pair=res,J;return pair
 result=least_squares(lambda x:evaluate(x)[0],np.clip(x0.ravel(),lo.ravel(),hi.ravel()),jac=lambda x:evaluate(x)[1],bounds=(lo.ravel(),hi.ravel()),tr_solver='lsmr',direct=True,max_nfev=25,ftol=1e-6,xtol=1e-7,gtol=1e-7)
 poses=[];err=[]
 for k,x in enumerate(result.x.reshape(count,nd)):
  q=bases[k].copy();delta=np.zeros(m.nv);delta[idx]=x;mujoco.mj_integratePos(m,q,delta,1.);poses.append(q)
  d.qpos[:]=q;mujoco.mj_kinematics(m,d);difference=kin.points()[0]-targets[k];difference[~supports[k],:2]=0.;difference[~supports[k],2]=np.minimum(difference[~supports[k],2],0.);err.append(float(np.max(np.linalg.norm(difference,axis=1))))
 q=poses[0].copy();xy,yaw,*_=spec.path(gait['period']);R=rotation_z(yaw);q[:3]=gait['origin']+R@(q[:3]-gait['origin'])+np.r_[xy,0.]
 rot=np.empty((3,3));mujoco.mju_quat2Mat(rot.ravel(),q[3:7]);mujoco.mju_mat2Quat(q[3:7],(R@rot).ravel());poses.append(q)
 return dict(gait,natural_clearance=True,configurations=np.array(poses),errors=err,closure_error=0.),dict(nfev=result.nfev,cost=result.cost,optimality=result.optimality)
