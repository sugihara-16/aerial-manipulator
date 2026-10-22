"""A zero-thrust trim must not create upward force when requesting descent."""
import copy
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.equilibrium import static_balance
from amsrr_det.flight import FlightController


def test_supplied_angular_acceleration_reaches_inverse_dynamics_request():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    rotation=pin.rpy.rpyToMatrix(.2,-.1,.4);q[3:7]=pin.Quaternion(rotation).coeffs()
    acceleration=np.zeros(model.nv);acceleration[3:6]=[.03,-.02,.01]
    flight=FlightController(model,meta,q)
    flight.update(q,np.zeros(model.nv),q,target_acceleration=acceleration)
    applied=np.array(flight.command_diagnostics['target_acceleration'])
    np.testing.assert_allclose(rotation@applied[3:6],acceleration[3:6],atol=1e-12)


def test_descent_allocation_respects_nonnegative_thrust_inside_the_solve():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    limited=copy.deepcopy(meta);limited['rotors'][0]['maximum']=0.
    trim=static_balance(model,limited,q)
    assert trim['status']=='static_screen_pass' and trim['thrust'][0]==0.
    target=q.copy();target[2]-=.01
    flight=FlightController(model,meta,q)
    _,forces=flight.update(q,np.zeros(model.nv),target,thrust_reference=np.array(trim['thrust']))
    assert np.all(forces>=0) and np.all(forces<=20)
    required_z=meta['mass']*(9.80665-4*.01)
    # All rotors initially point upward; clipping negative forces after solving
    # gives excess lift instead of this requested descent force.
    assert abs(sum(forces)-required_z)<1e-4


def test_vectoring_guard_limits_new_drive_request_before_physics_moves():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    target=q.copy();target[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.3,.2,0.)).coeffs()
    flight=FlightController(model,meta,q,guard_vectoring=True)
    command,forces=flight.update(q,np.zeros(model.nv),target)
    ids=flight.ai;effort=np.array([meta['actuators'][i]['effort'] for i in ids])
    raw=flight.kp[ids]*(command[ids]-q[flight.qi[ids]])
    assert np.all(abs(raw)<=effort+1e-9)
    assert flight.allocation_failure is None and flight.vectoring_guard_checks==1
    assert np.all(forces>=-1e-10) and np.all(forces<=20+1e-10)


def test_vectoring_guard_does_not_increase_an_existing_excess_request():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    flight=FlightController(model,meta,q,guard_vectoring=True)
    observed=q.copy();observed[flight.qi[flight.ai]]+=.05
    velocity=np.zeros(model.nv);velocity[flight.vi[flight.ai]]=.1
    before=flight.kp[flight.ai]*(flight.command[flight.ai]-observed[flight.qi[flight.ai]])-flight.kd[flight.ai]*.1
    command,_=flight.update(observed,velocity,q)
    raw=flight.kp[flight.ai]*(command[flight.ai]-observed[flight.qi[flight.ai]])-flight.kd[flight.ai]*.1
    effort=np.array([meta['actuators'][i]['effort'] for i in flight.ai])
    assert np.all(raw>=np.minimum(-effort,before)-1e-9)
    assert np.all(raw<=np.maximum(effort,before)+1e-9)
    assert flight.allocation_failure is None


def test_measured_thrust_uses_observed_directions_and_keeps_gimbal_update():
    from amsrr_det.rotor_distribution import rotor_actions
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    for a,angle in zip([a for a in meta['actuators'] if a['role']=='vectoring'],[.18,-.11,.23,-.16]):
        q[a['q']]=angle
    target=q.copy();target[:3]+=[.015,-.01,.002]
    target[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.02,-.01,.015)).coeffs()
    old=FlightController(model,meta,q,guard_vectoring=True)
    new=FlightController(model,meta,q,guard_vectoring=True,measured_thrust=True)
    previous,_=old.update(q,np.zeros(model.nv),target)
    command,forces=new.update(q,np.zeros(model.nv),target)
    np.testing.assert_array_equal(command[new.ai],previous[old.ai])
    # Independent generalized-force map at the observed pose. This root's
    # rotation is identity, so its first six rows are the world-frame wrench.
    action=rotor_actions(model,meta,q)[:6]
    diag=new.allocation_diagnostics[-1];residual=action@forces-np.array(diag['target'])
    np.testing.assert_allclose(action@forces,diag['measured_angle_action'],atol=1e-12)
    assert np.linalg.norm(residual)<np.linalg.norm(np.array(old.allocation_diagnostics[-1]['measured_angle_action'])-diag['target'])
    reference=np.full(new.nr,meta['mass']*9.80665/new.nr)
    gradient=action.T@residual+(.001/5.)**2*(forces-reference)
    # Optimality conditions for the actual force bounds, not just a solver flag.
    maximum=np.array([r['maximum'] for r in meta['rotors']])
    interior=(forces>1e-7)&(forces<maximum-1e-7)
    assert np.max(abs(gradient[interior]),initial=0.)<1e-8
    assert np.all(gradient[forces<=1e-7]>=-1e-8)
    assert np.all(gradient[forces>=maximum-1e-7]<=1e-8)


def test_measured_thrust_reports_unavailable_horizontal_force():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    target=q.copy();target[0]+=.1
    flight=FlightController(model,meta,q,guard_vectoring=True,measured_thrust=True)
    flight.update(q,np.zeros(model.nv),target)
    diag=flight.allocation_diagnostics[-1]
    assert diag['target'][0]>.1
    assert abs(diag['measured_angle_action'][0])<1e-12
    assert abs(diag['measured_angle_action'][0]-diag['target'][0])>.1


def test_failed_paired_allocation_keeps_last_issued_input(monkeypatch):
    import amsrr_det.rotor_distribution as allocation
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    flight=FlightController(model,meta,q,preserve_requested_action=True)
    previous_command,previous_force=flight.command.copy(),flight.force.copy()
    monkeypatch.setattr(allocation,'preserve_requested_action',
        lambda *args:(None,None,dict(accepted=False,reason='deliberate_test_rejection')))
    target=q.copy();target[2]+=.1
    command,force=flight.update(q,np.zeros(model.nv),target)
    assert flight.allocation_failure=='requested_action_allocation_not_found'
    np.testing.assert_array_equal(command,previous_command)
    np.testing.assert_array_equal(force,previous_force)
