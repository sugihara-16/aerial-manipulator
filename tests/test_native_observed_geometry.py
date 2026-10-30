import mujoco,numpy as np,pytest
from build.contact_step_control import pad_gap_jacobians,_python_pad_gap_jacobians

@pytest.mark.parametrize('distance',[.09,.2,1.1,2.5])
def test_batched_queries_keep_witness_direction_and_relative_jacobian(distance):
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="pad"><freejoint/><geom name="pad/one" type="sphere" size=".05"/>
      <geom name="pad/two" type="box" pos="0 .1 0" size=".02 .03 .04"/></body>
      <body name="object"><freejoint/><geom name="object" type="sphere" size=".05"/></body>
      </worldbody></mujoco>''');d=mujoco.MjData(m)
    d.qpos[7:10]=[distance,.013,.012];mujoco.mj_forward(m,d)
    actual=pad_gap_jacobians(m,d,['pad']);reference=_python_pad_gap_jacobians(m,d,['pad'])
    for a,b in zip(actual,reference):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
