"""Four causal comparisons from the identical acquired physical/controller state."""
import copy,json,time
from pathlib import Path
import numpy as np,pinocchio as pin,mujoco
from amsrr_det.grasp_task import plan_seed,GraspTask,smooth
from amsrr_det.objects import ObjectShape
from amsrr_det.task_audit import TaskAudit


def main():
 out=Path('artifacts/runs/ellipsoid_load_ablation');out.mkdir(parents=True,exist_ok=True)
 shape=ObjectShape('ellipsoid',[.3,.38,.24],.6);source=np.array([.745,0,.295,0,0,0,1.]);dest=source.copy();dest[0]+=.4
 plan=plan_seed(shape,source,dest,.175);p=plan.plant
 d=mujoco.MjData(p.mj);d.qpos[:]=p.configuration(plan.initial_q);mujoco.mj_forward(p.mj,d);task=GraspTask(plan)
 for i in range(6000):
  q,v=p.observation(d);cmd,f=task.update(i*.01,q,v);d.ctrl[:]=np.r_[cmd,f]
  for _ in range(10):mujoco.mj_step(p.mj,d)
  if task.stage=='LOAD':break
  if task.reason:raise RuntimeError(task.reason)
 else:raise RuntimeError('acquisition not reached')
 np.savez(out/'checkpoint.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time)
 results=[]
 for orientation,loading in [(0,0),(1,0),(0,1),(1,1)]:
  data=copy.deepcopy(d);flight=copy.deepcopy(task.flight);grasp=copy.deepcopy(task.grasp)
  audit=TaskAudit(p,['m1/yaw_dock_mech2','m2/yaw_dock_mech1'],shape.size,.175,shape=shape)
  records=[];start=time.perf_counter()
  for step in range(600):
   q,v=p.observation(data);t=step*.01;load=smooth(t/2) if loading else 0.
   target,tv=grasp.update(q,v,source,orientation_feedback=orientation)
   cmd,f=flight.update(q,v,target,target_velocity=tv,payload_mass=load*shape.mass,
       payload_com=q[p.meta['object']['q']:p.meta['object']['q']+3],
       joint_bias=(1-load)*task.j0+load*task.j1,thrust_reference=(1-load)*task.f0+load*task.f1,
       gimbal_reference=plan.closed_q[flight.qi[flight.ai]])
   data.ctrl[:]=np.r_[cmd,f]
   for _ in range(10):mujoco.mj_step(p.mj,data);audit.update(data)
   if step%5==0:
    body=p.mj.body('object').id;center=data.xpos[body];moments={k:np.zeros(3) for k in ['gripper','support','other']};forces={k:np.zeros(3) for k in moments}
    for cidx,c in enumerate(data.contact):
     names=[p.mj.geom(int(g)).name for g in [c.geom1,c.geom2]]
     if 'object' not in names:continue
     force=np.zeros(6);mujoco.mj_contactForce(p.mj,data,cidx,force)
     fworld=c.frame.reshape(3,3).T@force[:3]*(1 if names[1]=='object' else -1)
     other=names[1] if names[0]=='object' else names[0]
     group='gripper' if '/yaw_dock_mech' in other else 'support' if other in ['support','floor','destination_support'] else 'other'
     moments[group]+=np.cross(c.pos-center,fworld);forces[group]+=fworld
    q,v=p.observation(data);oq=p.meta['object']['q']
    records.append(dict(t=t,root=q[:7].tolist(),reference=target[:7].tolist(),object=q[oq:oq+7].tolist(),
       object_moments={k:v.tolist() for k,v in moments.items()},object_forces={k:v.tolist() for k,v in forces.items()},grasp=grasp.last_diagnostics))
   if audit.other_contacts or audit.max_force>30 or audit.max_joint_speed>3:break
  result=dict(orientation_feedback=orientation,load_transfer=loading,elapsed=step*.01,wall=time.perf_counter()-start,audit=audit.result(),trace=records)
  (out/f'orientation_{orientation}_load_{loading}.json').write_text(json.dumps(result,indent=2));results.append({k:v for k,v in result.items() if k!='trace'});print(json.dumps(results[-1]),flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
