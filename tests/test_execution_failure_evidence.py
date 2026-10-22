"""A failed geometric query must leave a failed result and replayable state."""
import json
from types import SimpleNamespace
import mujoco
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.mujoco_model import make_plant
from amsrr_det.objects import ObjectShape
from amsrr_det.clearance import MotionBoundedClearance
from scripts.run_grasp_task import execute


@pytest.mark.parametrize('where',['audit','controller'])
def test_geometry_failure_is_saved_without_acceptance(tmp_path,monkeypatch,where):
    shape=ObjectShape('box',[.3,.3,.3],.6)
    plant=make_plant(meshes=False,payload=shape.payload(),floor=True)
    q=pin.neutral(plant.pin);q[2]=1.
    oq=plant.meta['object']['q'];q[oq:oq+3]=[2.,0.,.15]
    pose=q[oq:oq+7].copy()
    trim=dict(thrust=[1.]*len(plant.meta['rotors']),
              joints=[dict(effort_nm=0.) for _ in plant.meta['actuators']])
    plan=SimpleNamespace(plant=plant,shape=shape,source=pose,destination=pose,support_top=0.,
        initial_q=q,open_q=q,closed_q=q,surfaces=[(0,'yaw1')],points=np.zeros((1,3)),
        normals=np.array([[1.,0.,0.]]),lift_height=.2,approach_height=.2,
        trims=[trim]*3,fit={},wall_s=0.,operating_point=None)
    def fail(*args,**kwargs):raise mujoco.FatalError('injected geometry failure')
    if where=='audit':monkeypatch.setattr(MotionBoundedClearance,'check',fail)
    else:
        from amsrr_det.grasp_task import GraspTask
        monkeypatch.setattr(GraspTask,'update',fail)
    result=execute(plan,tmp_path,{},reduced_physics=False)
    assert not result['task_success']
    assert result['reason'].startswith('numerical_failure:')
    assert 'injected geometry failure' in result['reason']
    assert json.loads((tmp_path/'result.json').read_text())['reason']==result['reason']
    state=np.load(tmp_path/'final_state.npz')
    assert np.isfinite(state['state']).all()
    interval=np.load(tmp_path/'last_issued_interval.npz')
    assert interval['duration_s']==(0. if where=='controller' else .01)
