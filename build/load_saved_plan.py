"""Development replay only; the task planner never reads this saved candidate."""
import json
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.model import build_model
from amsrr_det.mujoco_model import PlantModel
from amsrr_det.objects import ObjectShape
from amsrr_det.grasp_task import GraspPlan
from amsrr_det.grasp_control import surface_points


def load_saved_plan(directory):
    root=Path(directory)
    saved=json.loads((root/'plan.json').read_text());request=json.loads((root/'request.json').read_text())
    payload=request['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    m,meta=build_model(saved['model']['modules'],saved['model']['edges'],shape.payload())
    meta['material_mesh_cache']=saved['model'].get('material_mesh_cache')
    xml=(root/'scene.xml').read_text();mj=mujoco.MjModel.from_xml_string(xml)
    plant=PlantModel(m,meta,mj,xml,[(i,mj.joint(m.names[i]).id) for i in range(1,m.njoints)])
    samples=[surface_points(m,meta,np.array(saved['closed_q']),s) for s in saved['surfaces']]
    plan=GraspPlan(plant,shape,np.array(request['source']),np.array(request['destination']),request['support_top'],
        *[np.array(saved[k]) for k in ['open_q','closed_q','initial_q']],list(map(tuple,saved['surfaces'])),
        np.array(saved['points']),np.array(saved['normals']),samples,saved['trims'],saved['planning_time_s'],
        saved['fit'],saved['lift_height'],saved['approach_height'],saved.get('operating_point'))
    return plan,request
