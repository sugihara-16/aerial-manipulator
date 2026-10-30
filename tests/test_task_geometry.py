"""Physical reference contracts, independent of saved successful trajectories."""
import numpy as np
from scipy.spatial.transform import Rotation
from amsrr_det.objects import ObjectShape
from amsrr_det.task_geometry import GraspFrame, transport_height


def test_axisymmetric_frame_ignores_spin_but_follows_tilt():
    shape = ObjectShape('cylinder', [.2, .2, .4], 1.)
    initial = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    frame = GraspFrame(shape, initial)
    omega = np.zeros(3)
    spun = initial@Rotation.from_rotvec([0., 0., 1.2]).as_matrix()
    result, velocity, acceleration = frame.update(spun, omega, omega)
    np.testing.assert_allclose(result, initial, atol=1e-14)
    tilted = Rotation.from_rotvec([.3, .1, -.1]).as_matrix()@spun
    result, _, _ = frame.update(tilted, omega, omega)
    np.testing.assert_allclose(result[:, 2], tilted[:, 2], atol=1e-14)
    np.testing.assert_allclose(result.T@result, np.eye(3), atol=1e-14)


def test_axisymmetric_derivatives_match_rotating_axis():
    initial = Rotation.from_rotvec([.2, .4, -.1]).as_matrix()
    omega = np.array([.3, -.5, .2])
    alpha = np.array([.1, .2, -.3])
    shape = ObjectShape('cylinder', [.2, .2, .4], 1.)
    frame = GraspFrame(shape, initial)
    _, velocity, acceleration = frame.update(initial, omega, alpha)
    velocity = velocity.copy()
    acceleration = acceleration.copy()
    dt = 1e-6
    moved = Rotation.from_rotvec(omega*dt+.5*alpha*dt*dt).as_matrix()@initial
    _, next_velocity, _ = frame.update(moved, omega+dt*alpha, alpha)
    np.testing.assert_allclose((next_velocity-velocity)/dt, acceleration, atol=3e-7)
    np.testing.assert_allclose(np.cross(velocity, initial[:, 2]),
                               np.cross(omega, initial[:, 2]), atol=1e-14)


def test_only_exact_shape_symmetries_are_removed():
    assert not GraspFrame(ObjectShape('box', [.2, .2, .2], 1.), np.eye(3)).active
    assert not GraspFrame(ObjectShape('ellipsoid', [.2, .3, .4], 1.), np.eye(3)).active
    assert GraspFrame(ObjectShape('ellipsoid', [.2, .3, .3], 1.), np.eye(3)).axis == 0
    sphere = GraspFrame(ObjectShape('ellipsoid', [.2, .2, .2], 1.), np.eye(3))
    result, velocity, acceleration = sphere.update(
        Rotation.from_rotvec([.1, .2, .3]).as_matrix(), np.ones(3), np.ones(3))
    np.testing.assert_array_equal(result, np.eye(3))
    np.testing.assert_array_equal(velocity, np.zeros(3))
    np.testing.assert_array_equal(acceleration, np.zeros(3))


def test_transport_allowance_covers_bounded_pose_errors():
    rng = np.random.default_rng(168)
    for family in ['box', 'elongated_box', 'ellipsoid', 'cylinder', 'triangular_prism', 'tapered_box']:
        shape = ObjectShape(family, [.6, .6, 1.], 1.)
        height = transport_height(shape)
        for _ in range(30):
            rotation = Rotation.random(random_state=rng)
            pose = np.r_[np.zeros(3), rotation.as_quat()]
            pose[2] = -shape.lowest_z(pose)+height
            axis = rng.normal(size=3)
            axis *= .2/np.linalg.norm(axis)
            pose[3:] = (Rotation.from_rotvec(axis)*rotation).as_quat()
            pose[2] -= .03
            assert shape.lowest_z(pose) >= .1-1e-12
        # Increasing the height must not increase the old peak acceleration.
        assert np.isclose(height/(5*np.sqrt(height/.18))**2, .18/25)
