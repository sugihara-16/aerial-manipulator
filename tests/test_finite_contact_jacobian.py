"""Large structures trigger sparse contact geometry; preserve its action."""
import mujoco
import numpy as np
from build.finite_slip_preload import constraint_jacobian


def test_separated_pad_has_no_feasible_preload():
    from types import SimpleNamespace
    from build.sticking_design import LocalGrasp
    from build.finite_slip_preload import finite_slip_preload
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos="0 0 1"><joint name="hinge" type="hinge"/>
        <geom name="pad/surface" type="sphere" size=".1"/></body>
      <body pos="1 0 1"><freejoint name="object"/>
        <geom name="object" type="sphere" size=".1"/></body>
      </worldbody><actuator><motor joint="hinge"/></actuator></mujoco>''')
    plant = SimpleNamespace(mj=model, meta=dict(rotors=[], actuators=[
        dict(name='hinge', kp=200., kd=5., effort=1.3, role='dock')]))
    local = LocalGrasp(plant, model.qpos0, ['pad'])
    assert not local.elements
    solution, info = finite_slip_preload(local, model.qpos0, 23.)
    assert solution is None and not info['feasible']
    assert info['message'] == 'no active robot-object contacts'


def test_sparse_and_dense_contact_geometry_match_native_action():
    bodies = ''.join(f'<body pos="{i*3} 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>'
                     for i in range(11))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="100 100 .1"/>' + bodies + '</worldbody></mujoco>')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    assert model.nv > 60 and mujoco.mj_isSparse(model) and data.nefc > 0
    sparse = constraint_jacobian(model, data).copy()
    velocity = np.linspace(-.2, .3, model.nv); native = np.empty(data.nefc)
    mujoco.mj_mulJacVec(model, data, native, velocity)
    np.testing.assert_allclose(sparse @ velocity, native, atol=1e-13)
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    mujoco.mj_forward(model, data)
    assert not mujoco.mj_isSparse(model)
    np.testing.assert_allclose(constraint_jacobian(model, data), sparse, atol=1e-13)
