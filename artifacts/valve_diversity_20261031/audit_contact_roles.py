"""Offline sampled work audit: which robot surfaces turned which valve part."""
import json,sys,collections
from pathlib import Path
import numpy as np
import mujoco
root=Path(sys.argv[1]);stride=int(sys.argv[2]) if len(sys.argv)>2 else 10;results=[]
for case in json.loads((root/'cases.json').read_text()):
 folder=root/case['id'];rpath=folder/'result.json'
 if not rpath.exists():continue
 result=json.loads(rpath.read_text())
 if not result['task_success']:continue
 selection=json.loads((folder/'selection.json').read_text());surfaces=selection['plan']['surfaces']
 links=[f'm{n}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1]+'/' for n,port in surfaces]
 m=mujoco.MjModel.from_xml_path(str(folder/'physical.xml'))
 from amsrr_det.collision_policy import configure
 configure(m);m.opt.impratio=10000.;m.opt.ccd_tolerance=1e-12;m.opt.ccd_iterations=1000
 d=mujoco.MjData(m);obj=m.body('object').id
 from amsrr_det.grasp_geometry import grasp_frame
 sites=[m.site(grasp_frame(n,port)).id for n,port in surfaces]
 req=json.loads((folder/'request.json').read_text());axis=np.array(req['axis']);center=np.array(req['center'])
 states=np.load(folder/'physical_states.npz');trace=json.loads((folder/'physical_trace.json').read_text());force=np.zeros(6);work=collections.defaultdict(float);absolute=collections.defaultdict(float);samples=0;plane_work=0.;plane_distance_work=0.;plane_max=0.
 for k in range(0,len(trace),stride):
  row=trace[k]
  if row['phase']!='TURN':continue
  d.qpos[:]=states['q'][k];d.qvel[:]=states['v'][k];d.ctrl[:]=states['command'][k];mujoco.mj_forward(m,d);samples+=1
  for i,c in enumerate(d.contact):
   ids=list(map(int,c.geom));bodies=list(map(int,m.geom_bodyid[ids]))
   if obj not in bodies:continue
   oi=bodies.index(obj);other=1-oi;name=m.geom(ids[other]).name or ''
   if not name.startswith('m'):continue
   target=m.geom(ids[oi]).name or '';anchor=any(name.startswith(link)for link in links)
   group=('selected_dock' if anchor else 'other_robot')+('_bar' if target=='object' else '_wheel')
   mujoco.mj_contactForce(m,d,i,force);rotation=c.frame.reshape(3,3).T;sign=1. if oi==1 else -1.
   torque=axis@(np.cross(c.pos-center,sign*rotation@force[:3])+sign*rotation@force[3:])
   value=float(torque*row['speed']*.1*stride);work[group]+=value;absolute[group]+=abs(value)
   if anchor and target=='object':
    site=sites[next(j for j,link in enumerate(links) if name.startswith(link))]
    distance=abs(float((c.pos-d.site_xpos[site])@d.site_xmat[site].reshape(3,3)[:,0]))
    plane_distance_work+=distance*abs(value);plane_work+=abs(value);plane_max=max(plane_max,distance)
 row=dict(id=case['id'],modules=result['modules'],samples=samples,sampled_signed_work_j=dict(work),sampled_absolute_work_j=dict(absolute))
 total=sum(absolute.values());row['selected_dock_bar_absolute_fraction']=absolute['selected_dock_bar']/total if total else None
 row['anchor_plane_distance_m']=dict(work_weighted_mean=plane_distance_work/plane_work if plane_work else None,maximum=plane_max)
 results.append(row);print(case['id'],row['selected_dock_bar_absolute_fraction'],row['anchor_plane_distance_m'],flush=True)
(root/'contact_role_audit.json').write_text(json.dumps(dict(scope='Offline sampled contact work grouped by selected Dock link, not individual mesh triangles or continuous-time integral. Contact position distance to the authored anchor plane is also measured; these values are not controller inputs.',stride=stride,rows=results),indent=2))
