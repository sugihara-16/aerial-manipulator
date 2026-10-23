"""A proximity buffer must detect unsafe gaps without creating extra force."""
import mujoco
import numpy as np
from amsrr_det.clearance import detection_buffers,classify_contacts,proximity_model
from amsrr_det.clearance import MotionBoundedClearance,ContactClassifier


def test_geometry_query_keeps_all_contacts_without_unused_constraint_memory():
    from amsrr_det.clearance import forward_collision
    geoms=''.join('<geom type="sphere" size=".1" mass=".01"/>' for _ in range(32))
    xml='<mujoco><size memory="MEMORY"/><worldbody><body pos="0 0 1"><freejoint/>'+geoms+'</body><body pos=".19 0 1"><freejoint/>'+geoms+'</body></worldbody></mujoco>'
    limited=mujoco.MjModel.from_xml_string(xml.replace('MEMORY','1M'))
    full=mujoco.MjModel.from_xml_string(xml.replace('MEMORY','2M'))
    a,b,c=mujoco.MjData(limited),mujoco.MjData(limited),mujoco.MjData(full)
    warnings=[];previous=mujoco.get_mju_user_warning();mujoco.set_mju_user_warning(warnings.append)
    try:
        mujoco.mj_fwdPosition(limited,a)
        assert any('arena memory' in warning for warning in warnings)
        warnings.clear();forward_collision(limited,b);mujoco.mj_fwdPosition(full,c)
        assert not warnings
    finally:mujoco.set_mju_user_warning(previous)
    assert b.ncon==c.ncon==1024
    for name in ['geom1','geom2','dist','pos','frame']:
        np.testing.assert_array_equal(getattr(b.contact,name),getattr(c.contact,name))


def test_buffer_reports_near_environment_without_force_and_preserves_contact():
    xml='''<mujoco><option cone="elliptic"/><default><geom/></default><worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body pos="0 0 .12"><freejoint/><geom name="m0/body/0" type="sphere" size=".1" mass="1"/></body>
    <body pos="2 0 1"><freejoint/><geom name="object" type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>'''
    plain=mujoco.MjModel.from_xml_string(xml)
    buffered=mujoco.MjModel.from_xml_string(detection_buffers(xml))
    independent=proximity_model(plain)
    assert np.all(plain.geom_gap==0.)
    assert independent.geom_gap.max()==.03
    assert independent.opt.ccd_iterations>=500
    assert plain.opt.ccd_iterations<independent.opt.ccd_iterations
    independent.geom_gap[:]=.04
    assert np.all(plain.geom_gap==0.)
    a,b=mujoco.MjData(plain),mujoco.MjData(buffered)
    for height in [.12,.099]:
        a.qpos[2]=b.qpos[2]=height
        mujoco.mj_forward(plain,a);mujoco.mj_forward(buffered,b)
        np.testing.assert_allclose(a.qfrc_constraint,b.qfrc_constraint,atol=1e-12)
        names=[buffered.geom(i).name for i in range(buffered.ngeom)]
        intended,limits=classify_contacts(buffered,b,names,[])
        assert np.any(~intended & (np.asarray(b.contact.dist)<limits))
        if height>.1:
            assert a.ncon==0 and b.ncon==1 and b.contact[0].efc_address==-1


def test_proximity_refreshes_body_bounds_for_separated_boxes():
    xml='''<mujoco><default><geom/></default><worldbody>
    <body><geom name="support" type="box" size=".2 .2 .1"/></body>
    <body pos="0 0 .22"><freejoint/><geom name="m0/body/0" type="box" size=".1 .1 .1" mass="1"/></body>
    </worldbody></mujoco>'''
    plain=mujoco.MjModel.from_xml_string(xml)
    independent=proximity_model(plain)
    original=mujoco.MjData(plain);near=mujoco.MjData(independent)
    mujoco.mj_forward(plain,original);mujoco.mj_forward(independent,near)
    assert original.ncon==0
    assert near.ncon>0
    np.testing.assert_allclose(np.asarray(near.contact.dist),.02,atol=1e-12)
    np.testing.assert_allclose(near.qfrc_constraint,original.qfrc_constraint,atol=1e-12)
    assert independent.body_margin.max()==.03
    assert plain.body_margin.max()==0.


def test_query_filter_removes_only_always_permitted_support_pairs():
    xml='''<mujoco><default><geom/></default><worldbody>
    <geom name="support" type="box" size=".2 .2 .1"/>
    <body pos="0 0 .19"><freejoint/><geom name="object" type="box" size=".1 .1 .1" mass="1"/></body>
    <body pos=".19 0 .19"><freejoint/><geom name="robot" type="box" size=".1 .1 .1" mass="1"/></body>
    <body pos=".38 0 .19"><freejoint/><geom name="other" type="box" size=".1 .1 .1" mass="1"/></body>
    </worldbody></mujoco>'''
    plain=mujoco.MjModel.from_xml_string(xml)
    filtered=proximity_model(plain,omit_permitted_support=True)
    reference=mujoco.MjModel.from_xml_string(detection_buffers(xml))
    expected={};actual={}
    for model,found in [(reference,expected),(filtered,actual)]:
        data=mujoco.MjData(model);mujoco.mj_fwdPosition(model,data)
        for c in data.contact:
            pair=tuple(sorted([model.geom(c.geom1).name,model.geom(c.geom2).name]))
            found[pair]=min(found.get(pair,np.inf),c.dist)
    assert ('object','support') in expected and ('object','support') not in actual
    del expected[('object','support')]
    assert expected==actual
    assert ('object','robot') in actual and ('other','robot') in actual
    assert ('robot','support') in actual
    assert np.all(plain.geom_contype==1) and np.all(plain.geom_conaffinity==1)


def test_motion_bound_never_skips_translation_or_rotation_into_unsafe_gap():
    xml='''<mujoco><worldbody>
    <geom name="object" type="sphere" size=".01" pos=".1 .035 0"/>
    <body><freejoint/><geom name="arm/pad/0" type="box" size=".1 .01 .01" mass="1"/></body>
    </worldbody></mujoco>'''
    original=mujoco.MjModel.from_xml_string(xml)
    cached=MotionBoundedClearance(original,['arm/pad'])
    reference=proximity_model(original);data=mujoco.MjData(reference)
    classifier=ContactClassifier(reference,cached.names,[]);unsafe=0
    for axis in ['translation','rotation']:
        for amount in np.linspace(0,1,101):
            q=original.qpos0.copy()
            if axis=='translation':q[1]=.02*amount
            else:q[3:7]=[np.cos(.12*amount),0,0,np.sin(.12*amount)]
            quick,_,_=cached.check(q,False)
            data.qpos[:]=q;mujoco.mj_fwdPosition(reference,data)
            allowed,limits=classifier(data)
            exact=[tuple(sorted((cached.names[c.geom1],cached.names[c.geom2]))) for c,a,l in zip(data.contact,allowed,limits) if not a and c.dist<l-1e-6]
            assert set(quick)==set(exact)
            unsafe+=bool(exact)
    assert unsafe>0 and cached.reused>0
    # A phase change must invalidate even a physically motionless cache.
    assert cached.check(q,True)[0]==[]
    assert cached.check(q,False)[0]
    assert np.all(original.geom_gap==0.)


def test_joint_path_motion_bound_contains_independently_transformed_geom_points():
    xml='''<mujoco><worldbody><body><freejoint/>
      <geom name="object" type="sphere" size=".01" mass="1"/>
      <body pos=".2 0 0"><joint name="a" type="hinge" pos=".01 .02 0" axis="0 0 1"/>
        <geom type="box" size=".03 .02 .01" mass=".2"/>
        <body pos=".15 .03 0"><joint name="b" type="hinge" axis="0 1 0"/>
          <geom type="box" size=".1 .02 .04" pos=".05 .01 0" mass=".2"/>
        </body></body></body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);checker=MotionBoundedClearance(m,[])
    a,b=mujoco.MjData(m),mujoco.MjData(m);rng=np.random.default_rng(201)
    for _ in range(40):
        for d in [a,b]:
            d.qpos[:]=m.qpos0;d.qpos[:3]=rng.normal(0,.1,3)
            quat=rng.normal(size=4);d.qpos[3:7]=quat/np.linalg.norm(quat);d.qpos[7:]=rng.uniform(-1,1,2)
            mujoco.mj_kinematics(m,d)
        bound=checker.joint_motion.maximum(b.qpos,a.qpos)[0]
        # Use actual box corners, independently transformed by MuJoCo.
        for geom in range(m.ngeom):
            corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*m.geom_size[geom]
            pa=corners@a.geom_xmat[geom].reshape(3,3).T+a.geom_xpos[geom]
            pb=corners@b.geom_xmat[geom].reshape(3,3).T+b.geom_xpos[geom]
            assert np.linalg.norm(pb-pa,axis=1).max()<=bound+1e-12


def test_batched_clearance_checks_intermediate_poses_and_permission_changes():
    xml='''<mujoco><worldbody>
    <geom name="object" type="sphere" size=".01" pos=".1 .035 0"/>
    <body><freejoint/><geom name="arm/pad/0" type="box" size=".1 .01 .01" mass="1"/></body>
    </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    checker=MotionBoundedClearance(model,['arm/pad'])
    start=model.qpos0.copy();near=start.copy();near[1]=.00001
    assert not checker.check(start,False)[0]
    queries=checker.queries
    assert not checker.check_many([start,near,start],False)[0]
    assert checker.queries==queries
    collision=start.copy();collision[1]=.02
    assert checker.check_many([start,collision,start],False)[0]
    assert not checker.check_many([start,collision,start],True)[0]
    assert checker.check_many([start,collision,start],False)[0]
