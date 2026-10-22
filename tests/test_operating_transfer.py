from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from amsrr_det.operating_transfer import OperatingTransfer,pose
from amsrr_det.model import build_model
from amsrr_det.flight import FlightController


def test_load_entry_preserves_issued_commands_and_reference_velocity():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6));q=pin.neutral(model)
    flight=FlightController(model,meta,q);reference=q.copy()
    reference[flight.qi]+=.001;flight.command+=.002;flight.force+=.4
    nu=np.zeros(model.nv);nu[flight.vi]=.003
    dv=flight.vi[flight.dock]
    grasp=SimpleNamespace(model=model,reference=reference,vi=dv,geometry=lambda q:(None,None,None,np.array([.00013])))
    parameters=dict(preload=[.0002],joint_bias=np.zeros(len(flight.command)),thrust=flight.force+1,
        gimbal=flight.command[flight.ai]+.001,base_offset=[.002,0,0,0,0,.003])
    transfer=OperatingTransfer(parameters);command=flight.command.copy();force=flight.force.copy()
    transfer.start(grasp,flight,q,nu)
    preload,bias,thrust,gimbal=transfer.inputs(0.)
    reconstructed=reference[flight.qi]+(bias+flight.kd*nu[flight.vi])/flight.kp
    np.testing.assert_allclose(reconstructed,command,atol=1e-15)
    np.testing.assert_array_equal(flight.command,command);np.testing.assert_array_equal(flight.force,force)
    np.testing.assert_array_equal(grasp.last_velocity,np.r_[nu[:6],nu[dv]])
    np.testing.assert_array_equal(preload,[.00013]);np.testing.assert_array_equal(thrust,force)


def test_composed_rotation_offset_keeps_world_velocity_consistent():
    q=np.r_[np.zeros(3),pin.Quaternion.Identity().coeffs()]
    end=np.array([.01,0,0,0,0,.03]);transfer=OperatingTransfer(dict(base_offset=end))
    transfer.previous_offset=transfer.end_offset
    target=q.copy();target[3:7]=pin.Quaternion(pin.exp3(np.array([0.,0.,.001]))).coeffs()
    result,velocity=transfer.compose(q,q,target,np.r_[np.zeros(5),.1],1.,.01)
    # Rotating a translated reference offset contributes a linear velocity.
    expected=(pose(target)*transfer.end_offset).translation-transfer.end_offset.translation
    np.testing.assert_allclose(velocity[:3],expected/.01,atol=1e-14)
    assert np.linalg.norm(velocity[:3])>.0009
    np.testing.assert_allclose((pose(target).inverse()*pose(result)).homogeneous,transfer.end_offset.homogeneous,atol=1e-14)
