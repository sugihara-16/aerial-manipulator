"""Compare first contact against the source working surface, without control."""
import json,time
from pathlib import Path
import numpy as np,pinocchio as pin,mujoco
from amsrr_det.mujoco_model import make_plant
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.contact_geometry import close_material,material_triangles
from amsrr_det.objects import ObjectShape


def main():
 rows=[]
 for port in ['yaw1','pitch1']:
  for family,size in [('box',[.3,.4,.2]),('ellipsoid',[.3,.38,.24]),('cylinder',[.3,.3,.22])]:
   shape=ObjectShape(family,size,.6);p=make_plant(1,payload=shape.payload(),mesh_cache='build/collision_contacts',floor=False)
   q=pin.neutral(p.pin);q[2]=1.;d=p.pin.createData();pin.framesForwardKinematics(p.pin,d,q)
   f=p.meta['frames'][grasp_frame(0,port)];pad=d.oMf[f];oq=p.meta['object']['q'];q[oq+3:oq+7]=pin.Quaternion(pad.rotation).coeffs()
   object_rotation=pad.rotation;surface,_=shape.ray_surface([-1.,0,0]);base=pad.translation-object_rotation@surface
   # Moving the object along the pad's outward axis increases the gap.
   md=mujoco.MjData(p.mj);prefix='m0/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1]+'/'
   def gap(offset,kind):
    candidate=q.copy();candidate[oq:oq+3]=base+offset*pad.rotation[:,0]
    if kind=='source':
     obj=pin.SE3(object_rotation,candidate[oq:oq+3]);return close_material(shape,obj.inverse()*pad,material_triangles(port))[2].min()
    md.qpos[:]=p.configuration(candidate);mujoco.mj_forward(p.mj,md)
    gaps=[]
    for c in md.contact:
     names=[p.mj.geom(int(g)).name for g in [c.geom1,c.geom2]]
     if 'object' in names and any(name.startswith(prefix) for name in names):gaps.append(c.dist)
    return min(gaps) if gaps else .1
   onset=[]
   for kind in ['source','collision']:
    low,high=-.05,.02
    for _ in range(22):
     mid=(low+high)/2
     if gap(mid,kind)<0:low=mid
     else:high=mid
    onset.append((low+high)/2)
   row=dict(port=port,family=family,source_onset_m=onset[0],collision_onset_m=onset[1],difference_m=onset[1]-onset[0]);rows.append(row);print(json.dumps(row),flush=True)
 Path('artifacts/runs/contact_shape_onset.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
