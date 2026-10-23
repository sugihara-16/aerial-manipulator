import mujoco
import numpy as np
from amsrr_det.momentum_wrench import gravity_wrench, moving_origin_term
from build.momentum_feedback import rigid_motion_map


def test_moving_origin_balance_matches_independent_free_body_integration():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.0000001" gravity="0.3 -0.1 -9.81"/>
      <worldbody>
        <body name="payload" pos=".1 -.2 .5"><freejoint name="object"/>
          <geom type="box" size=".1 .2 .15" mass=".4"/></body>
        <body name="robot" pos=".5 .3 .8"><freejoint/>
          <geom type="ellipsoid" size=".1 .2 .3" pos=".1 0 0" mass="1.7"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = [.2, -.3, .1, .4, .2, -.1, -.3, .4, .2, -.1, .3, .2]
    mujoco.mj_forward(model, data)

    def momentum():
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        return rigid_motion_map(model, data.qpos).T @ mass @ data.qvel

    before = momentum()
    origin = data.qpos[:3].copy()
    transport = moving_origin_term(data.qvel[:3], before[:3])
    # Apply known world forces/moments at body centres of mass. These do not
    # use the actuator-map calculation being checked in the controller.
    data.xfrc_applied[1] = [1., -2., .5, .1, .2, -.3]
    data.xfrc_applied[2] = [-.2, .4, 1., -.3, .1, .2]
    applied = np.r_[data.xfrc_applied[:, :3].sum(axis=0),
        (data.xfrc_applied[:, 3:] + np.cross(data.xipos-origin,
                                           data.xfrc_applied[:, :3])).sum(axis=0)]
    predicted = applied + gravity_wrench(model, data, origin) - transport
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    measured = (momentum() - before) / model.opt.timestep
    np.testing.assert_allclose(measured, predicted, rtol=2e-6, atol=2e-6)


def test_gravity_moment_is_invariant_to_common_translation():
    from types import SimpleNamespace
    model = SimpleNamespace(body_mass=np.array([1., 2.]), opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])))
    points = np.array([[1., 2., 3.], [-.5, .1, .2]])
    origin = np.array([.3, .2, .1])
    shift = np.array([100., -200., 30.])
    a = gravity_wrench(model, SimpleNamespace(xipos=points), origin)
    b = gravity_wrench(model, SimpleNamespace(xipos=points+shift), origin+shift)
    np.testing.assert_allclose(a, b, atol=1e-12)
