from types import SimpleNamespace
import numpy as np
import pytest
import build.constrained_momentum_feedback as fallback
Controller=fallback.ConstrainedMomentumFeedback
from amsrr_det.path_progress import ScalarProgress

class PathProgress(ScalarProgress):
    def reference(self):return np.array([self.position]),np.array([self.speed])

def setup(monkeypatch,unresolved=False,status="PrimalInfeasible"):
    path=PathProgress(1.,10.,.0025);path.position=.2;path.speed=.03
    def fail(*args,**kwargs):
        path.advance(.01)
        raise ValueError('contact allocation '+status)
    monkeypatch.setattr(Controller,'_checked_update',fail)
    monkeypatch.setattr(fallback.mujoco,'mj_integratePos',lambda m,q,v,h:q.__iadd__(v*h))
    held=SimpleNamespace(air_gaps=np.zeros(1),model=None,horizon=.02,failures=0,correction=np.ones(1),calls=9)
    def correct(q,v,u):
        if unresolved:held.failures+=1
        return u+1.
    held.correct=correct
    fb=Controller.__new__(Controller)
    fb.contact_response=object();fb.previous_issued_input=np.array([.2]);fb.held_tracking=held;fb.m=SimpleNamespace(nv=1);fb.info={}
    return fb,path

def test_braking_reference_is_continuous_and_does_not_double_advance(monkeypatch):
    fb,path=setup(monkeypatch)
    before_position,before_speed=path.position,path.speed
    u=fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
    assert path.info['steps']==1
    assert path.speed==pytest.approx(before_speed-path.peak*path.period)
    assert path.position==pytest.approx(before_position+before_speed*path.period-.5*path.peak*path.period**2)
    assert path.speed>0
    assert u[0]==1.2 and fb.held_tracking.correction[0]==0
    assert fb.tracking_velocity[0]==path.speed

def test_failed_finite_response_never_returns_previous_input(monkeypatch):
    fb,path=setup(monkeypatch,True)
    with pytest.raises(ValueError,match='finite allocation fallback unresolved'):
        fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
    assert path.position==.2 and path.speed==.03 and path.info['steps']==0
    assert fb.info=={}

@pytest.mark.parametrize('status',['PrimalInfeasible','AlmostPrimalInfeasible','MaxIterations'])
def test_unresolved_allocation_statuses_reach_finite_response(monkeypatch,status):
    fb,path=setup(monkeypatch,status=status)
    fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
    assert fb.info['finite_allocation_fallbacks']==1

def test_path_snapshot_does_not_copy_the_physics_model(monkeypatch):
    class Model:
        def __deepcopy__(self,memo):raise AssertionError('model must remain shared')
    fb,path=setup(monkeypatch);path.model=Model()
    fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
    assert path.info['steps']==1


@pytest.mark.parametrize('unresolved',[False,True])
@pytest.mark.parametrize('reason',['bounded momentum fixed gimbal outside input domain','no attainable path acceleration','path cannot preserve nominal stopping domain'])
def test_fixed_gimbal_or_path_failure_uses_current_finite_response(monkeypatch,unresolved,reason):
    fb,path=setup(monkeypatch,unresolved)
    def fixed_failure(*args,**kwargs):
        path.advance(.01)
        raise ValueError(reason)
    monkeypatch.setattr(Controller,'_checked_update',fixed_failure)
    if unresolved:
        with pytest.raises(ValueError,match='finite allocation fallback unresolved'):
            fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
        assert path.position==.2 and path.speed==.03 and fb.info=={}
    else:
        command=fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
        assert command[0]==1.2 and path.info['steps']==1
        assert fb.info['finite_allocation_fallbacks']==1
        if reason in ('no attainable path acceleration','path cannot preserve nominal stopping domain'):
            assert fb.info['finite_path_braking_fallbacks']==1
            assert path.info['outside_nominal_stopping_domain_steps']==1

def test_unrelated_failure_does_not_enter_finite_recovery(monkeypatch):
    fb,path=setup(monkeypatch)
    def other_failure(*args,**kwargs):raise ValueError('bounded momentum wrench rank')
    monkeypatch.setattr(Controller,'_checked_update',other_failure)
    with pytest.raises(ValueError,match='bounded momentum wrench rank'):
        fb.update(np.zeros(1),np.zeros(1),np.zeros(1),path=path)
    assert fb.info=={}
