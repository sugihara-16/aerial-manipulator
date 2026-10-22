import numpy as np
import pytest
from amsrr_det.model import export_model
from amsrr_det.numerics import diagnostic_tolerances
from test_physics import execute


def test_adaptive_error_control_refines_without_changing_control_clock(tmp_path):
    model,meta=export_model(tmp_path)
    errors=[];steps=[]
    for scale in [10.,1.,.1]:
        scene=dict(duration_s=.1,integrator='adaptive_implicit_euler',thrust_lag_s=0.,step_tolerances=diagnostic_tolerances(scale))
        r=execute(tmp_path,model,meta,scene,.002)
        assert r['full_horizon']
        assert r['counts']['control']==10
        assert not r['task_accepted']
        errors.append(abs(r['final']['q'][2]-(1.+.5*model.gravity.linear[2]*.1**2)))
        steps.append(r['counts']['steps'])
    assert errors[0]>errors[1]>errors[2]
    assert steps[0]<steps[1]<steps[2]


def test_adaptive_contact_can_be_refined_without_relaxing_penetration_limit(tmp_path):
    from test_contact_integration import object_landing
    from amsrr_det.reference import integrate_reference
    model,meta=export_model(tmp_path,payload=dict(mass=1.,size=[.3,.4,.15]))
    scene=object_landing(model,meta)
    ref=integrate_reference(model,meta,scene,{},scene['duration_s'])
    ref_depth=float(max(.075-ref['trajectory'][meta['object']['q']+2]))
    scene.update(integrator='adaptive_implicit_euler',step_tolerances=diagnostic_tolerances())
    r=execute(tmp_path,model,meta,scene,.002)
    assert r['full_horizon']
    assert r['scenario']['max_penetration_m']==.002
    assert abs(r['metrics']['peak_penetration_m']-ref_depth)<1e-5
    assert r['counts']['rejected_steps']>0
