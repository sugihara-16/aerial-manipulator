"""Independent passive-axis physics and observation contracts."""

import numpy as np
import mujoco
import pytest
from amsrr_det.valve import ValvePlant, ValveSpec


@pytest.mark.parametrize('axis',[(0.,0.,1.),(1.,0.,0.),(0.,.6,.8)])
def test_bearing_restricts_motion_and_resists_both_rotation_directions(axis):
    spec=ValveSpec(axis=axis,torque=1.5)
    valve=ValvePlant(1,[],spec,meshes=False)
    m=valve.m
    assert m.nu==valve.control.mj.nu
    assert m.nv==valve.control.mj.nv-5
    for sign in [-1,1]:
        d=mujoco.MjData(m);d.qpos[2]=3.
        d.qfrc_applied[valve.ov]=sign*.8*spec.torque
        for _ in range(400):mujoco.mj_step(m,d)
        assert abs(d.qpos[valve.oi])<1e-5
        d.qfrc_applied[valve.ov]=sign*1.2*spec.torque
        for _ in range(400):mujoco.mj_step(m,d)
        assert sign*d.qpos[valve.oi]>.001
        np.testing.assert_allclose(d.xpos[m.body('object').id],spec.center,atol=1e-13)
        assert np.dot(d.xmat[m.body('object').id].reshape(3,3)[:,2],axis)>1-1e-12


def test_observation_embeds_hinge_pose_and_twist_without_force_truth():
    valve=ValvePlant(1,[],ValveSpec(axis=(0.,.6,.8),initial_angle=.3),meshes=False)
    d=mujoco.MjData(valve.m);d.qpos[valve.oi]=.7;d.qvel[valve.ov]=.4
    q,v=valve.observation(d)
    probe=mujoco.MjData(valve.control.mj);probe.qpos[:]=q;probe.qvel[:]=v
    mujoco.mj_forward(valve.m,d);mujoco.mj_forward(valve.control.mj,probe)
    a=valve.m.body('object').id;b=valve.control.mj.body('object').id
    np.testing.assert_allclose(d.xpos[a],probe.xpos[b],atol=1e-14)
    np.testing.assert_allclose(d.xmat[a],probe.xmat[b],atol=1e-14)
    sa=np.zeros(6);sb=np.zeros(6)
    mujoco.mj_objectVelocity(valve.m,d,mujoco.mjtObj.mjOBJ_BODY,a,sa,0)
    mujoco.mj_objectVelocity(valve.control.mj,probe,mujoco.mjtObj.mjOBJ_BODY,b,sb,0)
    np.testing.assert_allclose(sa,sb,atol=1e-14)


def test_valve_has_no_motor_and_physical_robot_limits_are_unchanged():
    valve=ValvePlant(1,[],ValveSpec(),meshes=False)
    assert all(valve.m.actuator(i).name!='object' for i in range(valve.m.nu))
    for field in ['actuator_forcerange','actuator_ctrlrange','actuator_gainprm']:
        np.testing.assert_array_equal(getattr(valve.m,field),getattr(valve.control.mj,field))
    assert valve.m.dof_frictionloss[valve.ov]==1.5


def test_rotation_axis_changes_port_generation_before_physical_screening():
    from amsrr_det.task_structure import grow_to_contacts
    points=np.array([[0.,0.,1.5]]);normal=np.array([[0.,1.,0.]])
    counts=[]
    for axis in [[0.,0.,1.],[1.,0.,0.]]:
        pitch=0
        for seed in range(64):
            _,surfaces,_,_,_=grow_to_contacts(1,points,normal,np.random.default_rng(seed),motion_axis=axis)
            pitch+=surfaces[0][1].startswith('pitch')
        counts.append(pitch)
    assert counts[1]>counts[0]


def test_task_local_model_cache_returns_independent_physical_models(tmp_path,monkeypatch):
    from amsrr_det import planning_memo
    monkeypatch.setattr(planning_memo,'active',planning_memo.PlanningMemo(tmp_path))
    first=ValvePlant(1,[],ValveSpec(),meshes=False)
    first.m.opt.gravity[0]=12.
    second=ValvePlant(1,[],ValveSpec(),meshes=False)
    assert second.m.opt.gravity[0]==0.
    first.configure_contacts(['m0/yaw_dock_mech1'])
    first.m.actuator_forcerange[0,1]=123.
    third=ValvePlant(1,[],ValveSpec(),meshes=False);third.configure_contacts(['m0/yaw_dock_mech1'])
    assert third.m.actuator_forcerange[0,1]!=123.


def test_in_place_coating_and_drives_match_independently_compiled_xml():
    from amsrr_det.collision_policy import configure
    valve=ValvePlant(2,[dict(a=0,port_a='pitch1',b=1,port_b='yaw2')],ValveSpec())
    valve.configure_contacts(['m0/yaw_dock_mech1','m1/pitch_dock_mech2'])
    for actual,xml in [(valve.m,valve.xml),(valve.control.mj,valve.control.xml)]:
        expected=mujoco.MjModel.from_xml_string(xml);configure(expected)
        expected.opt.impratio=10000.;expected.opt.ccd_tolerance=1e-12;expected.opt.ccd_iterations=1000
        for field in ['body_mass','body_inertia','body_invweight0','jnt_axis','jnt_range',
                      'actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange',
                      'geom_friction','geom_solref','geom_solimp','geom_priority']:
            np.testing.assert_array_equal(getattr(actual,field),getattr(expected,field))
        a,b=mujoco.MjData(actual),mujoco.MjData(expected)
        a.qpos[2]=b.qpos[2]=2.;a.ctrl[:]=b.ctrl[:]=.2
        for _ in range(20):mujoco.mj_step(actual,a);mujoco.mj_step(expected,b)
        np.testing.assert_array_equal(a.qpos,b.qpos)
        np.testing.assert_array_equal(a.qvel,b.qvel)


def test_geometry_symmetry_checks_actual_mount_and_dense_rigid_path():
    import json
    from pathlib import Path
    from amsrr_det.valve_plan import axis_symmetric_environment
    from amsrr_det.configuration_path import AxisRotationPath
    from amsrr_det.valve_search import links_for
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_yaw_symmetry_input.json').read_text())
    spec=ValveSpec(**saved['request']);system=ValvePlant(saved['modules'],saved['edges'],spec)
    links=links_for(saved['surfaces']);system.configure_contacts(links);p=system.control
    q=p.configuration(np.asarray(saved['q']));curve=AxisRotationPath(p,[0.,np.pi],q,spec.axis,spec.center)
    assert axis_symmetric_environment(p.mj,spec,q)
    d=mujoco.MjData(p.mj);depths=[];collisions=[]
    for angle in np.linspace(0,np.pi,181):
        d.qpos[:]=curve.pose(angle);mujoco.mj_fwdPosition(p.mj,d)
        depths.append(min((c.dist for c in d.contact),default=0.))
        forbidden=False
        for contact in d.contact:
            names=[p.mj.geom(int(g)).name for g in contact.geom]
            allowed='object' in names and any(any(n.startswith(link+'/') for link in links) for n in names)
            forbidden|=contact.dist<-.001 and not allowed
        collisions.append(forbidden)
    assert len(set(collisions))==1
    np.testing.assert_allclose(depths,depths[0],atol=2e-6)
    cylinder=p.mj.geom('valve_housing').id;p.mj.geom_pos[cylinder,0]+=.01
    assert not axis_symmetric_environment(p.mj,spec,q)
    p.mj.geom_pos[cylinder,0]-=.01;p.mj.geom_type[cylinder]=mujoco.mjtGeom.mjGEOM_BOX
    assert not axis_symmetric_environment(p.mj,spec,q)
