"""Physical geometry regressions, independent of robot control and task scores."""
import numpy as np
import mujoco
from amsrr_det.collision_policy import configure


def scene(kind):
    asset = ''
    if kind == 'prism':
        asset = '<asset><mesh name="prism" vertex="-.1 -.1 -.1 .1 -.1 -.1 0 .1 -.1 -.1 -.1 .1 .1 -.1 .1 0 .1 .1"/></asset>'
        geom, pose = 'type="mesh" mesh="prism"', 'pos="0 0 .0999995"'
    elif kind == 'cap':
        geom, pose = 'type="cylinder" size=".1 .2"', 'pos="0 0 .1999995"'
    else:
        geom, pose = 'type="cylinder" size=".1 .2"', 'pos="0 0 .0999995" quat=".7071067811865476 0 .7071067811865476 0"'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
        <option timestep=".000125" integrator="discrete" cone="elliptic" impratio="10000" ccd_tolerance="1e-12" ccd_iterations="1000"/>
        <default><geom friction=".6 0 0" condim="3" solref=".002 1" solimp=".95 .95 .001"/></default>
        {asset}<worldbody><geom name="support" type="box" size="1 1 .1" pos="0 0 -.1"/>
        <body name="object" {pose}><freejoint/><geom name="object" {geom} mass="1"/></body>
        </worldbody></mujoco>''')
    configure(model)
    return model, mujoco.MjData(model)


def test_flat_prism_support_dissipates_sliding_without_drift():
    model, data = scene('prism')
    data.qvel[0] = .004
    mujoco.mj_step(model, data, nstep=16000)
    assert np.linalg.norm(data.qvel) < 1e-5
    assert np.linalg.norm(data.qpos[:2]) < .001


def test_cylinder_cap_retains_face_support_and_stops():
    model, data = scene('cap')
    data.qvel[0] = .004
    mujoco.mj_integratePos(model, data.qpos, np.array([0.,0.,0.,.001,0.,0.]), 1.)
    mujoco.mj_step(model, data, nstep=16000)
    assert np.linalg.norm(data.qvel) < .003
    assert np.linalg.norm(data.qpos[:2]) < .001


def test_cylinder_side_normal_and_free_rolling_are_preserved():
    model, data = scene('side')
    data.qvel[1] = .01
    data.qvel[5] = -.1
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    assert abs(data.contact[0].frame[2]) > .999999
    configure(model)  # No wrapper recursion when configuring another private model.
    mujoco.mj_step(model, data, nstep=16000)
    assert abs(np.linalg.norm(data.qvel[:3])-.01) < .0001
