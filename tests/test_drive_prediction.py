import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.drive_prediction import DrivePrediction


@pytest.mark.parametrize('with_load',[False,True])
@pytest.mark.parametrize('offset',[.001,.2])
def test_coupled_prediction_obeys_inverse_dynamics_and_bounded_drive(offset,with_load):
    model,meta=build_model(2,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2')])
    rng=np.random.default_rng(552);q=pin.integrate(model,pin.neutral(model),rng.normal(0,.1,model.nv))
    v=rng.normal(0,.02,model.nv);force=rng.uniform(1.,7.,len(meta['rotors']));predictor=DrivePrediction(model,meta)
    command=q[predictor.qi]+offset*np.where(np.arange(len(predictor.qi))%2,1.,-1.)
    load=rng.normal(0,.1,model.nv) if with_load else np.zeros(model.nv)
    armature=model.armature.copy();qn,vn,info=predictor.step(q,v,command,force,generalized_load=load)
    external=[pin.Force.Zero() for _ in range(model.njoints)]
    for rotor,f in zip(meta['rotors'],force):
        frame=model.frames[rotor['frame']];external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
    required=pin.rnea(model,model.createData(),q,v,info['acceleration'],external)
    expected=np.zeros(model.nv);expected[predictor.vi]=np.clip(predictor.kp*(command-qn[predictor.qi])-predictor.kd*vn[predictor.vi],-predictor.limits,predictor.limits)
    np.testing.assert_allclose(required,expected+load,atol=1e-10)
    np.testing.assert_array_equal(model.armature,armature)
    np.testing.assert_array_equal(predictor.model.armature,armature)
    if offset==.2:assert np.any(np.isclose(abs(info['torque']),predictor.limits))


@pytest.mark.parametrize('with_load',[False,True])
@pytest.mark.parametrize('offset',[.001,.2])
def test_interval_sensitivity_matches_independent_command_perturbations(offset,with_load):
    model,meta=build_model(2,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2')]);rng=np.random.default_rng(334)
    q=pin.integrate(model,pin.neutral(model),rng.normal(0,.03,model.nv));v=rng.normal(0,.01,model.nv)
    prediction=DrivePrediction(model,meta);command=q[prediction.qi]+offset;force=np.full(len(meta['rotors']),4.)
    load=rng.normal(0,.1,model.nv) if with_load else np.zeros(model.nv)
    expected,ev,info=prediction.interval(q,v,command,force,linearize=True,generalized_load=load)
    inputs=np.r_[command,force];numerical_q=[];numerical_v=[];epsilon=1e-6
    for j in range(len(inputs)):
        plus=inputs.copy();minus=inputs.copy();plus[j]+=epsilon;minus[j]-=epsilon
        qp,vp,_=prediction.interval(q,v,plus[:len(command)],plus[len(command):],generalized_load=load)
        qm,vm,_=prediction.interval(q,v,minus[:len(command)],minus[len(command):],generalized_load=load)
        numerical_q.append((pin.difference(model,expected,qp)-pin.difference(model,expected,qm))/(2*epsilon))
        numerical_v.append((vp-vm)/(2*epsilon))
    np.testing.assert_allclose(info['position_wrt_input'],np.array(numerical_q).T,atol=1e-7,rtol=1e-5)
    np.testing.assert_allclose(info['velocity_wrt_input'],np.array(numerical_v).T,atol=1e-6,rtol=1e-5)
