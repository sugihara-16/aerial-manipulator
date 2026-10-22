import numpy as np
import pinocchio as pin
from scipy.linalg import null_space
from amsrr_det.model import build_model
from amsrr_det.flight import FlightController
from amsrr_det.paired_load import PairedLoadTransfer,unwind_saturated_request


def fixture():
    m,meta=build_model(3,[dict(a=0,port_a='yaw1',b=1,port_b='pitch2'),dict(a=0,port_a='yaw2',b=2,port_b='pitch1')])
    q=pin.neutral(m);q[2]=1.;flight=FlightController(m,meta,q);pair=PairedLoadTransfer(flight)
    E,J,_=pair.action(q,flight.command);basis=null_space(E,rcond=1e-9)
    _,_,right=np.linalg.svd(J@basis,full_matrices=False)
    delta=basis@right[0];delta*=1.3/max(abs(J@delta))
    motor=np.zeros(len(flight.actuators));motor[flight.dock]=J@delta
    raw=motor.copy();saturated=int(np.argmax(abs(motor)));raw[saturated]*=3
    command=q[flight.qi]+raw/flight.kp
    return flight,pair,q,command,np.full(flight.nr,5.),motor


def test_saturated_excess_is_removed_before_paired_static_action_transfer():
    f,pair,q,baseline,force,motor=fixture();v=np.zeros(f.model.nv)
    pair.capture(0.,q,v,baseline,force,motor);assert not pair.failure
    captured=pair.a.copy();previous=baseline.copy();E,J,_=pair.action(q,baseline)
    for k in range(340):
        command=baseline.copy();command[f.dock]+=pair.offset[f.dock]/f.kp[f.dock]
        result=pair.apply(k*.01,q,v,command,force,previous,motor)
        assert result is not None,pair.failure
        actual,thrust=result
        # An independent static position-drive calculation, including clipping.
        applied=np.clip(f.kp[f.dock]*(actual[f.dock]-q[f.qi[f.dock]]),-pair.limit,pair.limit)
        np.testing.assert_allclose(E@(thrust-force),0.,atol=1e-8)
        np.testing.assert_allclose(J@(thrust-force)+applied-motor[f.dock],0.,atol=1e-8)
        np.testing.assert_array_equal(pair.a,captured)
        previous=actual
    assert max(abs(applied)/pair.limit)<.01


def test_fixed_capture_does_not_relabel_new_saturation_as_feasible():
    f,pair,q,baseline,force,motor=fixture();v=np.zeros(f.model.nv)
    pair.capture(0.,q,v,baseline,force,motor)
    later=q.copy();index=np.flatnonzero(f.dock)[np.argmax(abs(motor[f.dock]))]
    later[f.qi[index]]-=np.sign(motor[index])*.01
    answer=pair.apply(pair.wind_time+.01,later,v,baseline,force,baseline,motor)
    assert answer is None and pair.failure=='captured_correction_did_not_clear_saturation'
    np.testing.assert_array_equal(pair.last_delta,0.)


def test_dynamic_unwind_tracks_moving_saturated_joint_without_cancelling_restoring_axis():
    kp=np.array([200.,200.]);kd=np.array([5.,5.]);limit=np.array([1.3,1.3])
    bounds=np.tile([-1.,1.],(2,1));v=np.array([.002,0.]);q=np.zeros(2)
    baseline=np.array([-.03,.002]);offset=np.zeros(2);previous=baseline.copy()
    motor=np.array([-1.3,.1]);step=np.full(2,.002125)
    for k in range(80):
        q[0]=k*.01*v[0]
        command=baseline+offset/kp
        revised,info=unwind_saturated_request(command,previous,q,v,motor,kp,kd,limit,bounds,step,.01)
        assert revised is not None
        assert abs(revised[0]-command[0])<=.001+1e-12
        assert revised[1]==baseline[1]
        offset+=kp*(revised-command);previous=revised
    np.testing.assert_allclose(kp[0]*(previous[0]-q[0])-kd[0]*v[0],-1.3,atol=1e-8)
    assert offset[0]>4.7


def test_dynamic_unwind_requires_measured_saturation_with_matching_sign():
    kp=np.full(2,200.);kd=np.full(2,5.);q=np.zeros(2);v=q.copy()
    command=np.array([-.03,-.03])
    result,info=unwind_saturated_request(command,command,q,v,np.array([-.5,1.3]),
        kp,kd,np.full(2,1.3),np.tile([-1.,1.],(2,1)),np.full(2,.002125),.01)
    np.testing.assert_array_equal(result,command)
    assert not any(info['active'])


def test_dynamic_unwind_then_paired_transfer_preserves_static_physical_action():
    f,_,q,baseline,force,motor=fixture();pair=PairedLoadTransfer(f,dynamic_unwind=True)
    v=np.zeros(f.model.nv);pair.capture(0.,q,v,baseline,force,motor)
    pair.transfer_deadline=6.;previous=baseline.copy();E,J,_=pair.action(q,baseline)
    frozen=None
    for k in range(340):
        command=baseline.copy();command[f.dock]+=pair.offset[f.dock]/f.kp[f.dock]
        result=pair.apply(k*.01,q,v,command,force,previous,motor)
        assert result is not None,pair.failure
        actual,thrust=result
        applied=np.clip(f.kp[f.dock]*(actual[f.dock]-q[f.qi[f.dock]]),-pair.limit,pair.limit)
        np.testing.assert_allclose(E@(thrust-force),0.,atol=1e-8)
        np.testing.assert_allclose(J@(thrust-force)+applied-motor[f.dock],0.,atol=1e-8)
        if pair.transfer_started is not None:
            if frozen is None:frozen=pair.a.copy()
            np.testing.assert_array_equal(pair.a,frozen)
        previous=actual
    assert pair.diagnostics['transfer_fraction']==1.
    assert max(abs(applied)/pair.limit)<.01


def test_saturation_history_does_not_require_the_entire_interval_to_be_saturated():
    kp=np.array([200.]);kd=np.array([5.]);q=np.zeros(1);v=q.copy()
    command=np.array([-1.302/200])
    result,info=unwind_saturated_request(command,command,q,v,np.array([-1.299]),
        kp,kd,np.array([1.3]),np.array([[-1.,1.]]),np.array([.002125]),.01,
        (np.array([-1.3]),np.array([-1.298])))
    assert info['active']==[True]
    np.testing.assert_allclose(200*result,-1.299,atol=1e-12)


def test_atomic_entry_issues_the_screened_pair_and_respects_progress_acceleration():
    f,_,q,baseline,force,motor=fixture();pair=PairedLoadTransfer(f,dynamic_unwind=True,atomic_start=True)
    v=np.zeros(f.model.nv);pair.capture(0.,q,v,baseline,force,motor);pair.transfer_deadline=6.
    previous=baseline.copy();E,J,_=pair.action(q,baseline);frozen=None;rate=0.;entered=False
    for k in range(340):
        command=baseline.copy();command[f.dock]+=pair.offset[f.dock]/f.kp[f.dock]
        result=pair.apply(k*.01,q,v,command,force,previous,motor)
        assert result is not None,pair.failure
        actual,thrust=result;applied=np.clip(f.kp[f.dock]*(actual[f.dock]-q[f.qi[f.dock]]),-pair.limit,pair.limit)
        np.testing.assert_allclose(E@(thrust-force),0.,atol=1e-8)
        np.testing.assert_allclose(J@(thrust-force)+applied-motor[f.dock],0.,atol=1e-8)
        assert abs(pair.progress_rate-rate)<=.01*10*np.sqrt(3)/3/4+1e-9
        rate=pair.progress_rate
        if pair.transfer_started is not None:
            if not entered:
                assert pair.progress>0. and np.max(abs(thrust-force))>0.
                frozen=pair.a.copy();entered=True
            np.testing.assert_array_equal(pair.a,frozen)
        previous=actual
    assert pair.progress==1. and rate<1e-8
