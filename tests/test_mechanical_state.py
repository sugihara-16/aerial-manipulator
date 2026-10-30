"""The reduced update must preserve full-model rigid dynamics at moving poses."""
import mujoco,numpy as np
from build.sticking_design import LocalGrasp


def test_mechanical_state_matches_full_dynamics_with_rotating_joints():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .15"><freejoint/><geom type="box" size=".1 .1 .1" mass="2"/>
        <body pos=".2 0 0"><joint axis="0 1 0" armature=".003" damping=".1"/>
          <geom type="box" size=".2 .02 .03" mass=".5"/>
        </body></body></worldbody></mujoco>''')
    local=LocalGrasp.__new__(LocalGrasp);local.m=model;local.d=mujoco.MjData(model)
    reference=mujoco.MjData(model);rng=np.random.default_rng(1027)
    for _ in range(20):
        q=model.qpos0.copy();mujoco.mj_integratePos(model,q,rng.normal(0,.2,model.nv),1.)
        v=rng.normal(0,.3,model.nv)
        reference.qpos[:]=q;reference.qvel[:]=v;mujoco.mj_forward(model,reference)
        local.mechanical_state(q,v)
        for field in ['qfrc_bias','xpos','xmat','site_xpos','subtree_com','cdof']:
            np.testing.assert_allclose(getattr(local.d,field),getattr(reference,field),rtol=1e-13,atol=1e-13)

        actual=np.empty((model.nv,model.nv));expected=np.empty_like(actual)
        mujoco.mj_fullM(model,local.d,actual);mujoco.mj_fullM(model,reference,expected)
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)


def test_material_values_equal_full_contact_coordinates():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body pos=".2 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
    local=LocalGrasp.__new__(LocalGrasp);local.m=model;local.d=mujoco.MjData(model)
    rng=np.random.default_rng(327)
    local.elements=[(0,1,2,rng.normal(size=3),rng.normal(size=3),np.eye(3))for _ in range(12)]
    for _ in range(20):
        q=model.qpos0.copy();mujoco.mj_integratePos(model,q,rng.normal(0,.2,model.nv),1.)
        local.at(q);gap,tangent,*_=local.contact()
        np.testing.assert_allclose(local.material_values(),np.c_[gap,tangent.reshape(-1,2)],rtol=1e-13,atol=1e-13)
