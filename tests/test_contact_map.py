import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.contact_map import prepare_contact_map,critical_damping_step


def geometry(points):
    model,meta=build_model(payload=dict(mass=1.,size=[.3,.4,.15]));q=pin.neutral(model)
    # Algebraic consistency fixture; not a claim these points are gripper pads.
    contacts=[dict(frame='m0/root',point=p,target_point=p,normal=[0.,0.,1.]) for p in points]
    return model,meta,q,contacts


def test_two_point_contacts_do_not_determine_full_object_orientation():
    model,meta,q,contacts=geometry([[-.2,0,0],[.2,0,0]])
    mapping=prepare_contact_map(model,meta,q,contacts)
    assert mapping.rank==5 and not mapping.supports_task_directions
    with pytest.raises(ValueError):mapping.correction(np.ones(6))


def test_noncollinear_points_recover_rigid_motion_and_virtual_work():
    model,meta,q,contacts=geometry([[-.2,-.1,0],[.2,-.1,0],[0,.2,0]])
    mapping=prepare_contact_map(model,meta,q,contacts)
    assert mapping.supports_task_directions
    rng=np.random.default_rng(48);velocity=rng.normal(size=mapping.robot_velocity.shape[1]);velocity[6:]=0.
    task,residual=mapping.compatible_velocity(velocity)
    np.testing.assert_allclose(task,velocity[:6],atol=1e-14)
    np.testing.assert_allclose(residual,0,atol=1e-14)
    wrench=rng.normal(size=6);loads=mapping.task_from_robot.T@wrench
    assert wrench@task==pytest.approx(loads@velocity,abs=1e-13)


def test_filter_keeps_position_velocity_pair_on_same_trajectory():
    z=np.array([.1,-.3]);v=np.array([.2,.05]);target=np.array([.4,-.2])
    end_z,end_v=critical_damping_step(z,v,target,10.,.01)
    middle=critical_damping_step(z,v,target,10.,.005)
    half_z,half_v=critical_damping_step(*middle,target,10.,.005)
    np.testing.assert_allclose(end_z,half_z,atol=1e-15)
    np.testing.assert_allclose(end_v,half_v,atol=1e-15)
    dz,dv=critical_damping_step(z,v,target,10.,1e-7)
    np.testing.assert_allclose((dz-z)/1e-7,v,atol=2e-6)
