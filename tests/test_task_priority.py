from types import SimpleNamespace
from build.task_priority import violation, secondary_violations


def audit(**changes):
    values=dict(other_contacts=set(),minimum_part_gap=.001,minimum_environment_gap=.03,
                max_joint_speed=1.,max_joint_limit_violation=0.,max_effort_ratio=1.,
                min_clearance_carried=.11,max_force=58.,max_moment=6.,max_penetration=.003,
                clearance_violations={'margin'},longest_gap=.08,max_material_displacement=.04)
    return SimpleNamespace(**(values|changes))


def test_auxiliary_exceedance_is_recorded_without_stopping():
    state=audit()
    assert violation(state) is None
    assert all(secondary_violations(state).values())


def test_native_audit_scalars_are_serializable():
    import json
    import numpy as np
    state=audit(max_force=np.float64(58.),max_joint_speed=np.float64(1.))
    assert json.loads(json.dumps(secondary_violations(state)))['contact_force'] is True


def test_drive_limit_collision_and_drop_still_fail():
    for changes,reason in [({'max_effort_ratio':1.01},'motor_effort_limit'),
                           ({'max_joint_limit_violation':.001},'joint_motion_limit'),
                           ({'max_joint_speed':3.1},'joint_motion_limit'),
                           ({'minimum_part_gap':-.001},'geometric_collision'),
                           ({'other_contacts':{'arm-object'}},'unplanned_collision'),
                           ({'min_clearance_carried':.05},'transport_clearance')]:
        assert violation(audit(**changes))==reason


def test_clearance_barriers_activate_at_contact_and_reuse_free_space():
    import mujoco
    import numpy as np
    from build.separation_rate_constraints import SeparationRateConstraints
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object"><freejoint/><geom name="object" type="sphere" size=".1"/></body>
      <body name="arm" pos=".4 0 0"><freejoint/><geom name="arm/geom" type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    rates=SeparationRateConstraints(model,[],all_geometry=True)
    rates.task_priority=True
    q=model.qpos0.copy();v=np.zeros(model.nv)
    assert len(rates.at(q,v)[0])==0
    queries=rates.query.queries
    assert len(rates.at(q,v)[0])==0
    assert rates.query.queries==queries and rates.query.reused>0
    q[7]=.2005
    assert len(rates.at(q,v)[0])==0  # The extra 1 mm margin is auxiliary.
    q[7]=.19999
    rows,bounds=rates.at(q,v)
    assert len(rows)>0
    original=SeparationRateConstraints(model,[],all_geometry=True)
    expected=original.at(q,v)
    np.testing.assert_allclose(rows,expected[0])
    np.testing.assert_allclose(bounds,expected[1])
