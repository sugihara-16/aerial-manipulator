import json
import numpy as np
from amsrr_det.model import export_model
from amsrr_det.diagnostics import fixture
from amsrr_det.benchmark import compare
from test_physics import execute
import pytest


def test_command_replay_keeps_same_control_ticks_for_half_step(tmp_path):
    model,meta=export_model(tmp_path)
    position=[0.]*len(meta['actuators']);velocity=position.copy()
    sc=dict(duration_s=.1,integrator='drive_implicit_euler',thrust_lag_s=.03,
            command_schedule=[dict(time=0.,thrust=[0.]*4,position=position,velocity=velocity),
                              dict(time=.05,thrust=[2.]*4,position=position,velocity=velocity)])
    results=[]
    for h in [.002,.001]:
        r=execute(tmp_path,model,meta,sc,h)
        assert r['full_horizon'];assert r['counts']['control']==10
        expected=2.*(1.-np.exp(-.05/.03))
        np.testing.assert_allclose(r['final']['f'],[expected]*4,atol=1e-13)
        results.append(r)
    assert results[0]['counts']['steps']==50
    assert results[1]['counts']['steps']==100


def test_observation_loss_does_not_produce_task_success(tmp_path):
    model,meta,sc=fixture(tmp_path)
    sc.update(duration_s=.1,observation_dropout_at_s=.03,integrator='drive_implicit_euler',
              thrust_lag_s=0.,gimbal_torque_lag_s=0.,dock_torque_lag_s=0.)
    r=execute(tmp_path,model,meta,sc,.002,closed=True)
    assert not r['task_accepted'] and not r['full_horizon']
    assert r['finite']
    assert r['stop_reason']=='stale observation; no validated recovery available'


def test_half_step_comparison_rejects_changed_physical_inputs(tmp_path):
    model,meta=export_model(tmp_path)
    a=execute(tmp_path,model,meta,dict(duration_s=.01),.001)
    b=json.loads(json.dumps(a));b['scenario']['thrust_lag_s']=.5
    with pytest.raises(ValueError):compare(model,a,b)
