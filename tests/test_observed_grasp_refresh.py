"""An unavailable optional refresh must retain the caller's valid reference."""
import importlib
from types import SimpleNamespace
import numpy as np
import mujoco
import pytest


def test_optional_observed_recovery_failure_is_reported_without_replacement(monkeypatch):
    from build.generated_grasp_stage import generate
    import amsrr_det.clearance as clearance
    finite=importlib.import_module('build.finite_slip_preload')
    stage=importlib.import_module('build.generated_grasp_stage')
    wrench=importlib.import_module('build.task_wrench_margin')
    model=SimpleNamespace(jnt_qposadr=np.array([0]),jnt_type=np.array([mujoco.mjtJoint.mjJNT_FREE]))
    q=np.array([0.,0.,1.,1.,0.,0.,0.])
    monkeypatch.setattr(stage,'LocalGrasp',lambda *args:None)
    monkeypatch.setattr(finite,'finite_slip_preload',lambda local,q,*args,**kwargs:((q.copy(),np.zeros(1),np.zeros(6)),dict(feasible=True)))
    monkeypatch.setattr(clearance,'MotionBoundedClearance',lambda *args:SimpleNamespace(check=lambda *args:(False,)))
    def unavailable(*args):
        error=ValueError('task recovery envelope not attained');error.evidence=dict(checked=True);raise error
    monkeypatch.setattr(wrench,'refine_task_reference',unavailable)
    args=(SimpleNamespace(mj=model),None,q,[],SimpleNamespace(source=np.zeros(3),destination=np.ones(3)))
    options=dict(finite_slip_preload=True,momentum_feedback=True,governed_motion=True,nearby_observation=True)
    answer=generate(*args,**options,fixed_observation=True)
    assert not answer['available']
    assert answer['refinement']['evidence']==dict(checked=True)
    assert 'goal' not in answer
    # Initial construction is mandatory: the same failure must still abort it.
    with pytest.raises(ValueError,match='recovery envelope'):
        generate(*args,**options,fixed_observation=False)
