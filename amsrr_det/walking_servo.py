"""Fixed position-servo tuning contrast, preserving the command interface.

ROBOTIS XC330 exposes Position PID gains. The existing Nm/rad parameters are
uncalibrated simulation control settings, not manufacturer specifications.
This comparison does not claim these numerical gains map to hardware register
values, and does not add a torque/velocity command to the gimbal interface.
"""
from dataclasses import replace
import copy
from lxml import etree as ET
import numpy as np
from build.mit_drive import dock_mit_model
from amsrr_det.mujoco_model import numbers

def walking_servo_model(plant,*,kp=0.,kd=.1):
 plant=dock_mit_model(plant,kp=kp,kd=kd);model=copy.copy(plant.mj);meta=copy.deepcopy(plant.meta);doc=ET.fromstring(plant.xml)
 for actuator in meta['actuators']:
  if actuator['role']=='dock':continue
  p,d=10.,.2
  node=doc.find('actuator').find(f"general[@name='{actuator['name']}']");node.set('gainprm',numbers([p]));node.set('biasprm',numbers([0,-p,-d]));actuator.update(kp=p,kd=d)
  i=model.actuator(actuator['name']).id;model.actuator_gainprm[i]=0.;model.actuator_gainprm[i,0]=p;model.actuator_biasprm[i]=0.;model.actuator_biasprm[i,:3]=[0.,-p,-d]
 for name in ['body_mass','body_inertia','dof_armature','jnt_range','actuator_forcerange','actuator_ctrlrange','geom_friction','geom_solref','geom_solimp']:
  if not np.array_equal(getattr(model,name),getattr(plant.mj,name)):raise ValueError('servo tuning altered '+name)
 return replace(plant,meta=meta,mj=model,xml=ET.tostring(doc,encoding='unicode'))
