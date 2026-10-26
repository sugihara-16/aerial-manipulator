import mujoco
import numpy as np
import pytest
from build.contact_step_control import pad_gap_jacobians

@pytest.mark.parametrize('distance',[.4,1.2,3.])
def test_separated_pad_gap_and_derivative(distance):
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><freejoint/><geom name="pad/a" type="sphere" size=".1"/></body>
    <body pos="2 0 0"><freejoint/><geom name="object" type="sphere" size=".2"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);d.qpos[7]=distance;mujoco.mj_forward(m,d)
    gaps,J=pad_gap_jacobians(m,d,['pad'])
    np.testing.assert_allclose(gaps,[distance-.3],atol=1e-12)
    np.testing.assert_allclose(J[0,[0,6]],[-1,1],atol=1e-12)
    h=1e-6;d.qpos[7]+=h;mujoco.mj_forward(m,d)
    later,_=pad_gap_jacobians(m,d,['pad'])
    np.testing.assert_allclose((later-gaps)/h,[J[0,6]],atol=1e-8)
