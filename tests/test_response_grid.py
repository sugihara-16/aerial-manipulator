"""The emitted hold must share the independent fine prediction's entire prefix."""
import copy
from types import SimpleNamespace
import mujoco,numpy as np
from build.finite_difference_response import FiniteDifferenceResponse


def predictors():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="pad"><joint name="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="pad/surface" type="box" size=".05 .05 .05" mass="1"/></body>
      <body pos="0 0 .099"><freejoint name="object"/>
        <geom name="object" type="box" size=".05 .05 .05" mass="1"/></body>
      </worldbody><actuator><general joint="hinge" gainprm="1" biasprm="0 -2 -1"
        forcerange="-5 5"/></actuator></mujoco>''')
    meta={'actuators':[dict(effort=5.)],'rotors':[]}
    coarse=SimpleNamespace(mj=copy.copy(model),meta=meta)
    fine=SimpleNamespace(mj=copy.copy(model),meta=meta)
    return (FiniteDifferenceResponse(coarse,['pad'],period=.02,steps=40,workers=1,
                fine_prefix_period=.0025,fine_timestep=.000125),
            FiniteDifferenceResponse(fine,['pad'],period=.0025,steps=20,workers=1))


def test_fine_prefix_matches_independent_hold_and_carries_state_into_tail():
    mixed,fine=predictors()
    try:
        q=mixed.m.qpos0.copy();v=np.zeros(mixed.m.nv);u=np.array([.03])
        a=mixed.evaluate(q,v,u);b=fine.evaluate(q,v,u)
        for name in ['states','loads','motors','penetration']:
            np.testing.assert_array_equal(a[name][:20],b[name])
        # Independently continue the fine integration state at coarse steps.
        data=mujoco.MjData(mixed.m);mujoco.mj_copyData(data,mixed.m,fine.thread.data)
        tail=mixed.runner.advance(data,35,include_penetration=True)
        for name,expected in zip(['states','loads','motors','penetration'],tail):
            np.testing.assert_array_equal(a[name][20:],expected)
        np.testing.assert_allclose(a['states'][:,0],np.r_[np.arange(1,21)*.000125,.0025+np.arange(1,36)*.0005],atol=1e-15)
        assert mixed.steps==55
    finally:mixed.close();fine.close()


def test_input_derivatives_use_the_same_complete_mixed_time_grid():
    mixed,fine=predictors()
    try:
        q=mixed.m.qpos0.copy();v=np.zeros(mixed.m.nv);u=np.array([.03])
        a=mixed.evaluate(q,v,u,True);b=fine.evaluate(q,v,u,True)
        assert a['derivatives'].shape==(55,2*mixed.m.nv,mixed.m.nu)
        np.testing.assert_array_equal(a['derivatives'][:20],b['derivatives'])
        np.testing.assert_array_equal(a['load_derivatives'][:20],b['load_derivatives'])
    finally:mixed.close();fine.close()
