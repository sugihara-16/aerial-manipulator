"""Isolated AK40-10 MIT-command model; not enabled in the task runner.

The scalar MuJoCo input encodes Kp*q_ref + Kd*v_ref + torque_ff.
Position and velocity terms still respond at every physics step, and the
SUM is clipped to the original continuous torque limit. Gimbals are unchanged.
The manufacturer-supported interface is documented in joint_actuators.yaml.
This is a nominal actuator model, not a measured hardware response model.
"""
from dataclasses import replace
import copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from amsrr_det.mujoco_model import numbers


def dock_mit_model(plant, *, kp=0., kd=5.):
    if not np.isfinite([kp,kd]).all() or not (0 <= kp <= 500 and 0 <= kd <= 5):
        raise ValueError('gains outside documented MIT ranges')
    doc=ET.fromstring(plant.xml)
    meta=copy.deepcopy(plant.meta)
    for actuator in meta['actuators']:
        if actuator['role']!='dock':continue
        node=doc.find('actuator').find(f"general[@name='{actuator['name']}']")
        node.set('gainprm','1')
        node.set('biasprm',numbers([0,-kp,-kd]))
        actuator.update(kp=float(kp),kd=float(kd),command_encoding='mit_combined')
    xml=ET.tostring(doc,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep=plant.mj.opt.timestep
    for name in ['nq','nv','nu']:
        if getattr(model,name)!=getattr(plant.mj,name):raise ValueError('drive change altered dimensions')
    for name in ['body_mass','body_inertia','dof_armature','jnt_range','actuator_forcerange']:
        if not np.array_equal(getattr(model,name),getattr(plant.mj,name)):
            raise ValueError('drive change altered physical parameter '+name)
    return replace(plant,meta=meta,mj=model,xml=xml)


def encode_commands(meta, position, velocity, torque_ff):
    """Position targets for gimbals; explicit MIT fields for Dock joints."""
    position,velocity,torque_ff=map(np.asarray,(position,velocity,torque_ff))
    n=len(meta['actuators'])
    if any(x.shape!=(n,) or not np.isfinite(x).all() for x in [position,velocity,torque_ff]):
        raise ValueError('one finite command per actuator is required')
    controls=position.copy()
    for i,a in enumerate(meta['actuators']):
        if a['role']=='dock':
            if a.get('command_encoding')!='mit_combined':raise ValueError('MIT model required')
            controls[i]=a['kp']*position[i]+a['kd']*velocity[i]+torque_ff[i]
        elif velocity[i]!=0 or torque_ff[i]!=0:
            raise ValueError('gimbal interface remains position-only')
    return controls
