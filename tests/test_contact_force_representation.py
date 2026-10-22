"""Verify the vectorized audit's force representation against MuJoCo's API."""
import mujoco
import numpy as np


def test_elliptic_constraint_multipliers_are_public_point_contact_forces():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option cone="elliptic" timestep=".001"/>
    <default><geom condim="3" friction=".8 .0 .0"/></default><worldbody>
    <geom type="plane" size="1 1 .1"/><body pos="0 0 .1005" euler="0 .01 .2">
    <freejoint/><geom type="box" size=".1 .1 .1" mass=".6"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model);data.qvel[:3]=[.03,-.01,-.01]
    checked=0
    for _ in range(100):
        mujoco.mj_step(model,data)
        for i,c in enumerate(data.contact):
            if c.efc_address<0:continue
            actual=np.zeros(6);mujoco.mj_contactForce(model,data,i,actual)
            local=data.efc_force[c.efc_address+np.arange(3)]
            np.testing.assert_allclose(local,actual[:3],atol=1e-12)
            np.testing.assert_array_equal(actual[3:],0.)
            assert local[0]>=0
            assert np.linalg.norm(local[1:])<=.8*local[0]+1e-10
            checked+=1
    assert checked>100
