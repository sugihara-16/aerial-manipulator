"""Use the planned articulated curve for airborne horizontal motion.
Stance stays on the fixed foothold; the same smooth clearance law remains.
No extra fitted swing curve can drift from the planned body/joint motion.
"""
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from .walking_cycle import CycleReference
from amsrr_det.walking import foot_link,rotation_z,foot_material

class KinematicReference(CycleReference):
 def __init__(self,plant,spec,gait):
  super().__init__(plant,spec,gait);self.plant=plant;self.d=mujoco.MjData(plant.mj);self.points=np.array(gait['foot_points']);self._at=None;self._rates=None
  # Contact labels belong to the generated structure.
  self.ids=[plant.mj.body(name).id for name in gait['foot_links']]
  self.natural_clearance=bool(gait.get('natural_clearance',False));self.material=foot_material(plant,gait['foot_links'])
  self.J=np.empty((3,plant.mj.nv));self.Jdot=self.J.copy();com=[];times=np.linspace(0,self.period,len(gait['configurations']))
  for t in times:
   self.d.qpos[:]=self.pose(t);mujoco.mj_kinematics(plant.mj,self.d);mujoco.mj_comPos(plant.mj,self.d);xy,yaw,*_=spec.path(t);com.append(rotation_z(-yaw)[:2,:2]@(self.d.subtree_com[0,:2]-self.origin[:2]-xy))
  com[-1]=com[0];self.com_curve=CubicSpline(times,com,bc_type='periodic')
 def rates(self,time):
  if self._at!=time:self._rates=super().rates(time);self._at=time
  return tuple(x.copy()for x in self._rates)
 def feet(self,time):
  p,v,a,active=super().feet(time);q,qd,qdd=self.rates(time);m=self.m;d=self.d;d.qpos[:]=q;d.qvel[:]=qd;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
  for i in np.flatnonzero(~active):
   bid=self.ids[i];point=d.xpos[bid]+d.xmat[bid].reshape(3,3)@self.points[i];mujoco.mj_jac(m,d,self.J,None,point,bid);mujoco.mj_jacDot(m,d,self.Jdot,None,point,bid);p[i,:2]=point[:2];v[i,:2]=(self.J@qd)[:2];a[i,:2]=(self.J@qdd+self.Jdot@qd)[:2]
   if self.natural_clearance:
    vertices=self.material[i][1];world=vertices@d.xmat[bid].reshape(3,3).T+d.xpos[bid];point=world[np.argmin(world[:,2])];mujoco.mj_jac(m,d,self.J,None,point,bid);mujoco.mj_jacDot(m,d,self.Jdot,None,point,bid);p[i,2]=point[2];v[i,2]=(self.J@qd)[2];a[i,2]=(self.J@qdd+self.Jdot@qd)[2]
  return p,v,a,active
 def com(self,time):
  t=time%self.period;xy,yaw,v,a=self.spec.path(time);R=rotation_z(yaw)[:2,:2];O=np.array([[0.,-1.],[1.,0.]])*self.spec.speed*self.spec.curvature;x=self.com_curve(t);d=self.com_curve(t,1);dd=self.com_curve(t,2)
  return self.origin[:2]+xy+R@x,v+R@(d+O@x),a+R@(dd+2*O@d+O@O@x)
