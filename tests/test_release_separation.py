"""Opening must protect unselected robot parts from the object as well as tables."""
import mujoco
import numpy as np
from build.separation_rate_constraints import SeparationRateConstraints


def scene():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1" pos="0 0 -2"/>
      <body pos="0 0 0"><freejoint/><geom name="m0/gimbal/0" type="sphere" size=".1"/></body>
      <body pos=".202 0 0"><freejoint name="object"/><geom name="object" type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')


def test_release_constrains_unselected_part_approach():
    m = scene()
    env = SeparationRateConstraints(m, [])
    whole = SeparationRateConstraints(m, [], all_geometry=True)
    q, v = m.qpos0.copy(), np.zeros(m.nv)
    assert env.at(q, v)[0].shape[0] == 0
    rows, lower = whole.at(q, v)
    assert len(rows) == 1
    np.testing.assert_allclose(lower, [-.002], atol=1e-10)
    closing = v.copy(); closing[6] = -.01
    assert (rows @ closing < lower).all()
    separating = -closing
    assert (rows @ separating > lower).all()


def test_intended_pad_remains_exempt():
    m = scene()
    whole = SeparationRateConstraints(m, ['m0/gimbal'], all_geometry=True)
    assert whole.at(m.qpos0, np.zeros(m.nv))[0].shape[0] == 0


def test_reused_distance_geometry_has_private_observations_and_same_constraints():
    m=scene();q=m.qpos0.copy();v=np.zeros(m.nv)
    original=SeparationRateConstraints(m,[],all_geometry=True)
    original.task_priority=True
    original.at(q,v)
    before=original.query.data.qpos.copy();gaps=original.query.model.geom_gap.copy()
    reused=SeparationRateConstraints(m,[],all_geometry=True,distance_query=original.query)
    reused.task_priority=True
    fresh=SeparationRateConstraints(m,[],all_geometry=True)
    fresh.task_priority=True
    assert reused.query.model is original.query.model
    assert reused.query.data is not original.query.data
    for shift in [0.,.001,.003,-.01]:
        moved=q.copy();moved[0]+=shift
        actual=reused.at(moved,v);expected=fresh.at(moved,v)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-12)
    np.testing.assert_array_equal(original.query.data.qpos,before)
    np.testing.assert_array_equal(original.query.model.geom_gap,gaps)
