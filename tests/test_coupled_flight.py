import numpy as np
import pinocchio as pin
import json
from pathlib import Path
from amsrr_det.model import build_model
from amsrr_det.equilibrium import static_balance
from amsrr_det.coupled_flight import CoupledFlightController,implicit_dock_velocity


def test_final_issued_command_is_the_rechecked_command_without_late_bias():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    trim=static_balance(model,meta,q);control=CoupledFlightController(model,meta,q)
    target=q.copy();target[:3]+=[.005,-.005,.001];v=np.zeros(model.nv)
    previous=control.command.copy()
    command,force=control.update(q,v,target,joint_bias=np.array([j['effort_nm'] for j in trim['joints']]),thrust_reference=np.array(trim['thrust']))
    assert control.allocation_failure is None
    json.dumps(control.command_diagnostics)  # Long runs must retain their terminal evidence.
    assert np.all(abs(command-previous)<=control.speed*control.dt+1e-7)
    predicted,velocity,info=control.predictor.interval(q,v,command,force,history=True)
    np.testing.assert_allclose(predicted,control.coupled_diagnostics['predicted_q'],atol=1e-13)
    np.testing.assert_allclose(velocity,control.coupled_diagnostics['predicted_v'],atol=1e-13)
    for retained, repeated in zip(control.prediction_samples, info['samples']):
        np.testing.assert_allclose(retained['q'], repeated['q'], atol=1e-13)
        np.testing.assert_allclose(retained['v'], repeated['v'], atol=1e-13)
    for sample in info['samples']:
        assert np.all(abs(sample['v'][control.vi])<=control.speed+1e-7)
        assert np.all(sample['q'][control.qi]>=control.limits[:,0]-1e-7)
        assert np.all(sample['q'][control.qi]<=control.limits[:,1]+1e-7)


def test_implicit_dock_reference_dissipates_energy_with_small_coupled_inertia():
    model,meta=build_model(2,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2')]);q=pin.neutral(model)
    dock=[a for a in meta['actuators'] if a['role']=='dock'];ids=[a['v'] for a in dock]
    mass=pin.crba(model,model.createData(),q)[np.ix_(ids,ids)];n=len(ids);rng=np.random.default_rng(74)
    kp=np.array([a['kp'] for a in dock]);kd=np.array([a['kd'] for a in dock]);dt=.01
    for _ in range(20):
        error=rng.normal(0,.01,n);velocity=rng.normal(0,.1,n)
        nxt=implicit_dock_velocity(mass,kp,kd,error,velocity,np.zeros(n),np.zeros(n),dt)
        energy=lambda e,v:.5*v@mass@v+.5*np.sum(kp*e**2)
        assert energy(error-dt*nxt,nxt)<energy(error,velocity)
    # Independent linear transition check: continuous PD gains do not make its
    # explicit 10 ms velocity target stable for this low-inertia mechanism.
    A=-dt*np.linalg.solve(mass,np.diag(kp));B=np.eye(n)-dt*np.linalg.solve(mass,np.diag(kd))
    assert max(abs(np.linalg.eigvals(np.block([[np.eye(n)+dt*A,dt*B],[A,B]]))))>1


def test_planned_input_center_removes_ineffective_saturated_position_excess():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    trim=static_balance(model,meta,q);bias=np.array([j['effort_nm'] for j in trim['joints']]);force=np.array(trim['thrust'])
    controls=[CoupledFlightController(model,meta,q,track_vectoring=True,planned_input_reference=use_plan) for use_plan in [False,True]]
    selected=int(np.flatnonzero(controls[0].dock)[0]);before=q[controls[0].qi].copy();before[selected]+=.2
    output=[]
    for control in controls:
        control.command=before.copy();control.force=force.copy();control.control_count=1
        command,thrust=control.update(q,np.zeros(model.nv),q,joint_bias=bias,thrust_reference=force)
        assert control.allocation_failure is None
        assert np.all(abs(command-before)<=control.speed*control.dt+1e-7)
        assert np.all(thrust>=0) and np.all(thrust<=20.)
        output.append(command)
    assert output[1][selected]<output[0][selected]-.01
    # Both remain deeply saturated. The change removes ineffective input,
    # without pretending that raw position-drive demand equals applied torque.
    assert output[1][selected]-q[controls[1].qi[selected]]>.1


def test_saturation_crossing_rejects_full_step_and_returns_feasible_descent():
    fixture=json.loads((Path(__file__).parent/'fixtures/coupled_drive_saturation.json').read_text())
    model,meta=build_model(fixture['model']['modules'],fixture['model']['edges']);q=np.array(fixture['q']);v=np.array(fixture['v'])
    control=CoupledFlightController(model,meta,q)
    control.command=np.array(fixture['previous_command']);control.force=np.array(fixture['previous_force']);control.control_count=1
    command,force=control.update(q,v,np.array(fixture['target']))
    assert control.allocation_failure is None
    report=control.coupled_diagnostics
    assert report['iterations'][0]['trials'][0]['nonlinear_cost']>report['initial_cost']*100
    assert report['cost']<report['initial_cost']
    expected,velocity,info=control.predictor.interval(q,v,command,force,history=True)
    np.testing.assert_allclose(expected,report['predicted_q'],atol=1e-13)
    np.testing.assert_allclose(velocity,report['predicted_v'],atol=1e-13)
    for sample in info['samples']:
        assert np.all(abs(sample['v'][control.vi])<=control.speed)
        assert np.all(sample['q'][control.qi]>=control.limits[:,0])
        assert np.all(sample['q'][control.qi]<=control.limits[:,1])


def test_vectoring_reference_is_tracked_with_all_physical_drives():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    trim=static_balance(model,meta,q);control=CoupledFlightController(model,meta,q,track_vectoring=True)
    control.force=np.array(trim['thrust']);control.control_count=1
    target=q.copy();actuator=meta['actuators'][control.ai[0]];target[actuator['q']]+=.01
    command,force=control.update(q,np.zeros(model.nv),target)
    assert control.allocation_failure is None
    assert len(control.coupled_diagnostics['desired_tracking_velocity'])==6+len(meta['actuators'])
    # Reintegrate the actual final held input. A gimbal target must cause motion
    # toward that target rather than being absent from the tracked state.
    _,velocity,info=control.predictor.interval(q,np.zeros(model.nv),command,force,history=True)
    assert velocity[actuator['v']]>1e-3
    assert all(np.all(abs(row['v'][control.vi])<=control.speed+1e-7) for row in info['samples'])


def test_all_joint_implicit_reference_dissipates_energy():
    model,meta=build_model(2,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2')]);q=pin.neutral(model)
    indices=[a['v'] for a in meta['actuators']];mass=pin.crba(model,model.createData(),q)[np.ix_(indices,indices)]
    kp=np.array([a['kp'] for a in meta['actuators']]);kd=np.array([a['kd'] for a in meta['actuators']])
    rng=np.random.default_rng(48)
    for _ in range(20):
        error=rng.normal(0,.01,len(indices));v=rng.normal(0,.1,len(indices))
        nxt=implicit_dock_velocity(mass,kp,kd,error,v,np.zeros(len(indices)),np.zeros(len(indices)),.01)
        initial=.5*v@mass@v+.5*np.sum(kp*error**2)
        final=.5*nxt@mass@nxt+.5*np.sum(kp*(error-.01*nxt)**2)
        assert final<initial


def test_full_joint_target_preserves_stationary_and_constant_speed_tracking():
    model,meta=build_model(2,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2')])
    indices=[a['v'] for a in meta['actuators']]
    mass=pin.crba(model,model.createData(),pin.neutral(model))[np.ix_(indices,indices)]
    kp=np.array([a['kp'] for a in meta['actuators']]);kd=np.array([a['kd'] for a in meta['actuators']])
    for nu in [np.zeros(len(indices)),np.linspace(-.08,.08,len(indices))]:
        # At time k, q=r_k and v=nu. The end-time reference is r_k+dt*nu.
        actual=implicit_dock_velocity(mass,kp,kd,.01*nu,nu,nu,np.zeros_like(nu),.01)
        np.testing.assert_allclose(actual,nu,atol=1e-14)


def test_quintic_world_derivatives_match_independent_position_differences():
    from amsrr_det.numerics import quintic_reference
    model,meta=build_model();first=pin.neutral(model)
    first[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.3,-.4,.7)).coeffs()
    delta=np.linspace(-.1,.2,model.nv);last=pin.integrate(model,first,delta)
    t=.7;h=1e-5;duration=2.3
    q,v,a,_=quintic_reference(model,meta,first,last,t,duration)
    minus,vm,_,_=quintic_reference(model,meta,first,last,t-h,duration)
    plus,vp,_,_=quintic_reference(model,meta,first,last,t+h,duration)
    np.testing.assert_allclose(v[:3],(plus[:3]-minus[:3])/(2*h),atol=1e-9)
    rm=pin.Quaternion(minus[3:7]).matrix();rp=pin.Quaternion(plus[3:7]).matrix()
    np.testing.assert_allclose(v[3:6],pin.log3(rp@rm.T)/(2*h),atol=1e-9)
    np.testing.assert_allclose(a,(vp-vm)/(2*h),atol=1e-8)
    for elapsed in [-1,0,duration,duration+1]:
        _,v,a,_=quintic_reference(model,meta,first,last,elapsed,duration)
        np.testing.assert_array_equal(v,0);np.testing.assert_array_equal(a,0)


def test_base_model_defect_is_measured_at_next_observation_in_world_axes_once():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.1,-.2,.4)).coeffs()
    control=CoupledFlightController(model,meta,q,track_vectoring=True,base_residual_feedback=True)
    control.update(q,np.zeros(model.nv),q)
    assert control.allocation_failure is None
    np.testing.assert_array_equal(control.base_prediction_bias,0.)
    first=control.coupled_diagnostics
    q1=np.array(first['predicted_q']);v1=np.array(first['predicted_v'])
    innovation=np.array([.0001,-.0002,.0003,.0004,-.0005,.0006])
    R=pin.Quaternion(q1[3:7]).matrix()
    v1[:3]+=R.T@innovation[:3];v1[3:6]+=R.T@innovation[3:]
    control.update(q1,v1,q)
    assert control.allocation_failure is None
    info=control.coupled_diagnostics['base_residual_feedback']
    np.testing.assert_allclose(info['innovation'],innovation,atol=1e-14)
    np.testing.assert_allclose(info['estimate'],control.base_bias_alpha*innovation,atol=1e-14)
    np.testing.assert_allclose(np.array(info['nominal_velocity_goal'])-info['corrected_velocity_goal'],info['estimate'],atol=1e-14)
    # The next stored prediction must remain the uncorrected physical rollout.
    np.testing.assert_allclose(control.last_predicted_base,control.coupled_diagnostics['predicted_tracking_velocity'][:6],atol=1e-14)
