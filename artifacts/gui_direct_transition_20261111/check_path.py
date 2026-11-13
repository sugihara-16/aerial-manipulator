from pathlib import Path
import json,numpy as np,mujoco,pinocchio as pin
from build.load_saved_plan import load_saved_plan
from amsrr_det.task_sequence import Coordinates
from amsrr_det.collision_policy import configure
p=next(Path('artifacts/gui_direct_transition_20261111/grasp_direct/runs').iterdir())
plan,_=load_saved_plan(p/'execution/generated_plan');plant=plan.plant
physical=mujoco.MjModel.from_xml_path(str(p/'execution/physical.xml'))
z=np.load(p/'assembly/final.npz');print('finalkeys',z.files)
d=mujoco.MjData(plant.mj);d.qpos[:],d.qvel[:]=Coordinates(physical,plant.mj).observe(z['q'],z['v']);start,_=plant.observation(d);goal=plan.open_q.copy()
configure(plant.mj)
print('root startgoal',start[:3],goal[:3])
for delay in [0.,.2,.4,.6]:
 bad=[]
 for t in np.linspace(0,1,41):
  f=lambda x:np.clip(x,0,1)**3*(10-15*np.clip(x,0,1)+6*np.clip(x,0,1)**2)
  u=f(t/(1-delay/2));q=pin.interpolate(plant.pin,start,goal,u);q[:3]=(1-u)*start[:3]+u*goal[:3]
  q[2]=start[2]+f((t-delay)/(1-delay))*(goal[2]-start[2])
  d.qpos[:]=plant.configuration(q);mujoco.mj_kinematics(plant.mj,d);mujoco.mj_collision(plant.mj,d)
  pairs=[]
  for c in d.contact:
   n=[plant.mj.geom(int(g)).name for g in c.geom]
   if c.dist<-.0005 and any(x.startswith('m') for x in n):pairs.append((round(float(c.dist),4),n))
  if pairs:bad.append((round(t,3),pairs[:2]))
 print('delay',delay,'bad',bad)
