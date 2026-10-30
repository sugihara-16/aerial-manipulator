"""Physical input/load contracts for the new object-relative controller."""
from types import SimpleNamespace
import numpy as np
import mujoco
from build.relative_task_control import RobotDrive, DesiredContactLoad, ObjectRelativeControl


def plant():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep=".000125" gravity="0 0 -9.80665"/>
      <worldbody>
        <body pos="-.0999 0 1"><freejoint name="base0"/>
          <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
          <site name="rotor"/>
          <body name="pad"><joint name="yaw" range="-1 1" axis="0 0 1" armature=".01"/>
            <geom name="pad/sphere" type="sphere" size=".05" mass=".1"/>
          </body>
        </body>
        <body name="payload_mass" pos="0 0 1"><freejoint name="object"/>
          <inertial mass="2" pos="0 0 0" diaginertia=".01 .01 .01"/>
          <body name="object"><geom name="object" type="sphere" size=".05" mass="0"/></body>
        </body>
      </worldbody>
      <actuator>
        <general name="yaw" joint="yaw" gainprm="1" biasprm="0 -200 -5" biastype="affine" forcerange="-1.3 1.3"/>
        <general site="rotor" gear="0 0 1 0 0 0" ctrlrange="0 50"/>
      </actuator>
    </mujoco>''')
    meta = dict(actuators=[dict(name='yaw', role='dock', kp=200., kd=5., effort=1.3, lower=-1., upper=1.)],
                rotors=[dict(frame=0, minimum=0., maximum=50.)])
    return SimpleNamespace(mj=model, meta=meta, pin=SimpleNamespace(frames=[SimpleNamespace(name='rotor')]))


def test_payload_load_uses_inertial_parent_and_never_an_unactuated_object_input():
    p = plant()
    drive = RobotDrive(p)
    q = p.mj.qpos0.copy()
    load = DesiredContactLoad(p, q, ['pad'])
    assert p.mj.body_mass[load.object_body] == 0.
    assert load.mass == 2.
    drive.observe(q, np.zeros(p.mj.nv))
    wanted = np.array([0., 0., 19.6133, 0., 0., 0.])
    c = load.prepare(drive, wanted, active_only=True)
    assert len(c['cols']) > 0
    np.testing.assert_array_equal(c['reaction'][drive.ov:drive.ov+6], 0.)
    np.testing.assert_array_equal(drive.B[drive.ov:drive.ov+6], 0.)


def test_separated_pad_cannot_supply_a_fictitious_payload_force():
    p = plant()
    drive = RobotDrive(p)
    q = p.mj.qpos0.copy()
    load = DesiredContactLoad(p, q, ['pad'])
    q[0] -= .1
    drive.observe(q, np.zeros(p.mj.nv))
    wanted = np.array([0., 0., 19.6133, 0., 0., 0.])
    c = load.prepare(drive, wanted, active_only=True)
    assert len(c['cols']) == 0
    assert not c['active'].any()


def test_drive_preserves_real_effort_and_thrust_bounds_under_large_requests():
    p = plant()
    drive = RobotDrive(p)
    q = p.mj.qpos0.copy()
    v = np.zeros(p.mj.nv)
    drive.observe(q, v)
    controller = ObjectRelativeControl(p, q, ['pad'])
    for requested in [-1e3, 1e3]:
        target = q[drive.oi:drive.oi+7].copy()
        target[:3] += requested
        command = controller.update(q, v, q, target)
        torque = command[0]-200*q[drive.qi[0]]-5*v[drive.vi[0]]
        assert abs(torque) <= 1.30000001
        assert 0. <= command[1] <= 50.


def test_gravity_compensation_excludes_free_payload_before_contact():
    p = plant()
    drive = RobotDrive(p)
    q = p.mj.qpos0.copy()
    drive.observe(q, np.zeros(p.mj.nv))
    controller = ObjectRelativeControl(p, q, ['pad'])
    q[0] -= .1
    command = controller.update(q, np.zeros(p.mj.nv), q, q[drive.oi:drive.oi+7], payload_fraction=0.)
    # Only 1.1 kg of robot weight is a robot load before grasping.
    np.testing.assert_allclose(command[1], 1.1*9.80665, atol=1e-4)


def test_top_face_at_box_edge_is_not_reinterpreted_as_sideways_pressure():
    from amsrr_det.objects import ObjectShape
    p = plant()
    drive = RobotDrive(p)
    q = p.mj.qpos0.copy()
    load = DesiredContactLoad(p, q, ['pad'])
    drive.observe(q, np.zeros(p.mj.nv))
    pad, bp, bo, _, _, _ = load.elements[0]
    # This pad's task is to press down on the top edge. The closest vector
    # from a slightly outside point happens to be horizontal at that edge.
    point = np.array([0., .05001, 1.0499])
    frame = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    load.elements = [(pad, bp, bo, point-drive.d.xpos[bp],
                      point-drive.d.xpos[bo], frame)]
    load.shape = ObjectShape('box', [.1, .1, .1], 2.)
    load.planar = True
    values = load.prepare(drive, np.zeros(6), active_only=True)
    np.testing.assert_allclose(values['G'][:3, 0], [0., 0., -1.])
    np.testing.assert_allclose(values['gaps'], [1e-5], atol=1e-12)


def test_soft_contact_action_reaction_has_no_net_internal_wrench():
    from build.relative_task_control import CurrentContactLoad
    from build.momentum_feedback import rigid_motion_map
    p = plant()
    drive = RobotDrive(p)
    q = p.mj.qpos0.copy()
    q[0] += .001  # finite penetration makes distinct surface arms observable
    load = CurrentContactLoad(p, q, ['pad'])
    drive.observe(q, np.zeros(p.mj.nv))
    c = load.prepare(drive, np.zeros(6), active_only=True)
    assert len(c['cols'])
    np.testing.assert_allclose(
        (rigid_motion_map(p.mj, q).T@c['reaction'])[:, c['cols']],
        c['G'][:, c['cols']], atol=1e-14)


def test_symmetric_grasp_frame_keeps_full_object_orientation_task():
    from amsrr_det.objects import ObjectShape
    p = plant()
    closed = p.mj.qpos0.copy()
    shape = ObjectShape('ellipsoid', [.1, .1, .1], 2.)
    controller = ObjectRelativeControl(p, closed, ['pad'], object_shape=shape)
    q = closed.copy()
    oi = controller.drive.oi
    q[oi+3:oi+7] = [np.cos(.15), 0., 0., np.sin(.15)]
    controller.update(q, np.zeros(p.mj.nv), closed, closed[oi:oi+7],
                      payload_fraction=1., track_object=True)
    # Rotation of a sphere changes no geometric grasp reference, but the
    # full object orientation error must still demand a correcting torque.
    np.testing.assert_allclose(controller.delta[:6], 0., atol=1e-14)
    assert np.linalg.norm(controller.last['requested_object_wrench'][3:]) > .01
