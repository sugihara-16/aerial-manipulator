from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from build.held_input_guard import HeldInputGuard


def guard_for(loads):
    guard = HeldInputGuard.__new__(HeldInputGuard)
    response = dict(states=np.zeros((3, 3)), loads=loads,
                    motors=np.zeros((3, 2, 1)), penetration=np.zeros((3, 1)))
    guard.repairs = 0
    guard.control = SimpleNamespace(m=SimpleNamespace(nq=1), nj=1,
        qi=np.array([0]),vi=np.array([0]),qlo=np.array([-1.]),qhi=np.array([1.]),
        effort=np.array([1.]),links=['pad'],
        clearance=SimpleNamespace(check_many=lambda *args:([],None,None)),
        predictor=SimpleNamespace(evaluate=lambda *args:response),
        update=Mock(return_value=np.array([.2])))
    return guard


def test_release_repair_detects_mid_interval_peak_not_only_endpoint():
    loads = np.zeros((3, 1, 7)); loads[1, 0, 1] = 31.
    guard = guard_for(loads)
    np.testing.assert_array_equal(guard.update(np.zeros(1),np.zeros(1),np.array([.3])),[.2])
    assert guard.repairs == 1
    guard.control.update.assert_called_once()
    np.testing.assert_array_equal(guard.control.previous,[.2])


def test_valid_release_keeps_original_input():
    loads = np.zeros((3, 1, 7)); loads[:, 0, 1] = 10.
    guard = guard_for(loads)
    np.testing.assert_array_equal(guard.update(np.zeros(1),np.zeros(1),np.array([.3])),[.3])
    assert guard.repairs == 0
    guard.control.update.assert_not_called()
    np.testing.assert_array_equal(guard.control.previous,[.3])


def test_recovery_checks_vector_speed_even_when_each_component_is_below_limit():
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="object"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    guard=guard_for(np.zeros((3,1,7)))
    guard.control.m=model
    guard.control.object_velocity_norm_limits=(.02,.1)
    velocity=np.zeros(model.nv);velocity[:2]=[.0198,-.0035]
    assert abs(velocity).max()<.02 and np.linalg.norm(velocity)>.02
    states=np.tile(np.r_[.005,model.qpos0,velocity],(3,1))
    guard.control.predictor.evaluate=lambda *args:dict(states=states,
        loads=np.zeros((3,1,7)),motors=np.zeros((3,2,1)),penetration=np.zeros((3,1)))
    guard.update(model.qpos0,np.zeros(model.nv),np.array([.3]))
    guard.control.update.assert_called_once()


def test_object_motion_checks_pose_and_velocity_separately():
    import mujoco
    from build.held_input_guard import object_response_in_tracking_domain
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="object"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    q=model.qpos0.copy();v=np.zeros(model.nv)
    def allowed(position,speed):
        predicted=q.copy();predicted[0]=position
        predicted_v=v.copy();predicted_v[0]=speed
        state=np.r_[.0025,predicted,predicted_v][None]
        return object_response_in_tracking_domain(model,q,v,q,v,state)
    assert allowed(.02,.01)
    assert not allowed(.031,0.)
    assert not allowed(.02,-.04)  # v + 2*position cancels; separate bounds must not.


def test_object_motion_follows_reference_and_does_not_increase_existing_excess():
    import mujoco
    from build.held_input_guard import object_response_in_tracking_domain
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="object"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    reference=model.qpos0.copy();velocity=np.zeros(model.nv);velocity[0]=.1
    q=reference.copy();q[0]=.04
    for position,expected in [(.03925,True),(.04125,False)]:
        predicted=q.copy();predicted[0]=position
        state=np.r_[.0025,predicted,velocity][None]
        assert object_response_in_tracking_domain(model,q,velocity,reference,velocity,state)==expected


def test_object_check_uses_the_repaired_final_input_response(monkeypatch):
    import build.held_input_guard as guards
    loads=np.zeros((3,1,7));loads[1,0,1]=31.
    guard=guard_for(loads);final_states=np.ones((3,3))
    guard.control.response={'states':final_states}
    seen=[]
    import build.object_tracking_domain as domains
    monkeypatch.setattr(domains,'ObjectTrackingDomain',lambda *args:object())
    monkeypatch.setattr(guards,'object_response_in_tracking_domain',
        lambda model,q,v,reference,velocity,states:seen.append(states) or False)
    import pytest
    with pytest.raises(ValueError,match='object tracking domain'):
        guard.update(np.zeros(1),np.zeros(1),np.array([.3]),reference=np.zeros(1),velocity=np.zeros(1))
    assert len(seen)==1 and seen[0] is final_states


def test_unresolved_friction_uses_validated_response_and_retains_limits(monkeypatch):
    import build.contact_unload as release
    inverse = SimpleNamespace(objective='input',previous=None,update=Mock(return_value=np.array([.2])))
    state = release.ContactUnload.__new__(release.ContactUnload)
    state.release = None; state.first = np.array([.3]); state.elapsed = 0.
    state.local = None; state.q = np.r_[np.zeros(3),1.,np.zeros(3)]; state.oi = 0
    state.links = ['pad']; state.open_start = np.zeros(1)
    state.control = SimpleNamespace(m=None,d=None,groups=None,
        update=Mock(side_effect=ValueError('contact solve PrimalInfeasible')))
    state.guard = SimpleNamespace(control=inverse,repairs=0,update=lambda q,v,u:u)
    state.response_fallbacks = 0
    monkeypatch.setattr(release,'gravity_command',lambda *args:(np.array([.3]),None))
    monkeypatch.setattr(release,'pad_gap_jacobians',lambda *args:(np.array([-.0001]),None))
    command = state.update(state.q,np.zeros(6),np.array([.3]))
    np.testing.assert_array_equal(command,[.2])
    assert inverse.objective == 'input'
    assert state.response_fallbacks == 1
    assert state.release is None  # A computed input is not separation evidence.


def test_withdrawal_keeps_checking_after_the_opening_controller_hands_off(monkeypatch):
    from build.contact_unload import ContactUnload
    import build.held_input_guard as guards
    calls = []
    guard = SimpleNamespace(repairs=1, update=lambda q,v,u:u*.5)
    monkeypatch.setattr(guards, 'HeldInputGuard',
        lambda plant,links,model,period:calls.append(period) or guard)
    state = ContactUnload.__new__(ContactUnload)
    state.release = SimpleNamespace(update=lambda *a:np.array([.6]), ready=True,
                                    complete=False, info={})
    state.withdrawal_guard = None
    state.plant = None; state.links = []
    state.plan = SimpleNamespace(plant=SimpleNamespace(mj=None))
    np.testing.assert_allclose(state.update(None,None,None), [.3])
    np.testing.assert_allclose(state.update(None,None,None), [.3])
    assert calls == [.01]
    assert state.info['withdrawal_hold_repairs'] == 1


def test_tracking_violation_alone_enters_the_constrained_input_repair(monkeypatch):
    import build.held_input_guard as guards
    import build.object_tracking_domain as domains
    guard=guard_for(np.zeros((3,1,7)));domain=object()
    monkeypatch.setattr(domains,'ObjectTrackingDomain',lambda *args:domain)
    accepted=iter([False,True])
    monkeypatch.setattr(guards,'object_response_in_tracking_domain',lambda *args:next(accepted))
    guard.control.response={'states':np.ones((3,3))}
    output=guard.update(np.zeros(1),np.zeros(1),np.array([.3]),reference=np.zeros(1),velocity=np.zeros(1))
    np.testing.assert_array_equal(output,[.2])
    guard.control.update.assert_called_once()
    assert guard.control.object_tracking_domain is domain
