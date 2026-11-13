"""The common input must influence execution, reject ignored requests, and preserve physics."""
import copy
import json
import mmap
from types import SimpleNamespace
import numpy as np
import pytest
from amsrr_det.task_spec import defaults,make_spec,validate_spec,form_values,requirements
from amsrr_det.task_execution import compile_request

@pytest.mark.parametrize('kind',['grasp','valve','walking'])
@pytest.mark.parametrize('mode',['task','assembly','integrated'])
def test_round_trip(kind,mode):
    spec=make_spec(kind,defaults(kind),mode)
    assert validate_spec(json.loads(json.dumps(spec)))==spec
    assert form_values(spec)==defaults(kind)
    r=requirements(spec)
    assert r['resources']==spec['robot']
    assert r['contact_domain']==spec['contacts']
    assert ('assemble' in r['stages'])==(mode!='task')
    assert r['motion']==spec['goal']

@pytest.mark.parametrize('path,value',[
    (('goal','speed_m_s'),float('nan')),(('robot','module_count','fixed'),True),
    (('execution','delay'),'ideal'),(('execution','control_period_s'),.1),
    (('scene','gravity_m_s2'),[0,0,-1]),(('version',),True),
    (('scene','obstacles'),[]),(('contacts','count','max'),20),
    (('goal','initial_xy_m'),[1,0]),(('robot','module_count','fixed'),2)])
def test_unsupported_conditions_fail_explicitly(path,value):
    spec=make_spec('walking',defaults('walking'))
    d=spec
    for k in path[:-1]:d=d[k]
    d[path[-1]]=value
    with pytest.raises(ValueError):validate_spec(spec)

def test_grasp_input_reaches_physical_request():
    v=defaults('grasp');v.update(mass=2.,sx=.7,x=1.2,y=-.4,roll=15.,yaw=33.,dx=.6,dy=.2)
    r,request=compile_request(make_spec('grasp',v))
    assert request['payload']['mass']==2.
    assert request['payload']['size'][0]==.7
    np.testing.assert_allclose(np.array(request['destination'])[:3]-request['source'][:3],[.6,.2,0])
    np.testing.assert_allclose(request['source'][:2],[1.2,-.4])
    assert request['source'][2]>.2
    assert r['derived']['gravity_force_n'][2]==-2*9.80665

def test_valve_axis_and_goal_are_consumed():
    v=defaults('valve');v.update(tilt=90.,azimuth=90.,torque=12.,angle=60.,x=2.5,y=1.5)
    r,q=compile_request(make_spec('valve',v))
    np.testing.assert_allclose(q['axis'],[0,1,0],atol=1e-12)
    assert q['torque']==12. and q['target_angle']==np.pi
    assert q['center']==(2.5,1.5,1.5)
    assert q['initial_angle']==pytest.approx(np.pi/3)
    assert r['contact_domain']['allowed_region']==['local_y_positive','local_y_negative']

def test_walk_fixed_count_and_curvature_are_consumed():
    v=defaults('walking');v.update(modules=7,speed=.3,curvature=-.3,distance=7.)
    r,q=compile_request(make_spec('walking',v))
    assert q==dict(modules=7,speed=.3,distance=7.,curvature=-.3)
    assert r['derived']['lateral_acceleration_m_s2']==pytest.approx(-.027)

def test_live_state_is_a_read_only_copy(tmp_path):
    from amsrr_det.task_live import LiveState,snapshot
    pub=LiveState(tmp_path);pub.begin(SimpleNamespace(nq=2,nv=1,neq=1),'model')
    d=SimpleNamespace(qpos=np.array([1.,2.]),qvel=np.array([3.]),eq_active=np.array([True]),time=4.)
    pub.update(d,'TURN')
    with open(tmp_path/'live.bin','rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        t,phase,state=snapshot(mm,{})
        assert t==4. and phase=='TURN'
        np.testing.assert_array_equal(state,[1.,2.,3.,1.]);state[:]=0
        np.testing.assert_array_equal(d.qpos,[1.,2.])
    pub.close()

def test_projection_keeps_real_secondary_base_unmodified():
    import mujoco
    from amsrr_det.task_sequence import Coordinates
    # Reduced controller has one base. Physical model retains the second free base.
    physical=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="base"/><geom size=".1"/><body><joint name="hinge"/><geom size=".1" pos=".2 0 0"/></body></body><body pos="1 0 0"><freejoint name="other"/><geom size=".1"/></body></worldbody></mujoco>')
    reduced=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="base"/><geom size=".1"/><body><joint name="hinge"/><geom size=".1" pos=".2 0 0"/></body></body></worldbody></mujoco>')
    d=mujoco.MjData(physical);d.qpos[-7]=1.3;d.qvel[-6]=.4
    before=d.qpos.copy();q,v=Coordinates(physical,reduced).observe(d.qpos,d.qvel)
    assert q.shape==(8,) and v.shape==(7,)
    np.testing.assert_array_equal(d.qpos,before)
    assert d.qvel[-6]==.4

def test_output_switch_starts_at_applied_input():
    from amsrr_det.task_sequence import BlendedControl
    class Control:
        last={}
        def update(self,*a,**k):return np.array([2.,3.])
    c=BlendedControl(Control(),np.array([.2,.3]))
    np.testing.assert_allclose(c.update(observation_time=10.),[.2,.3])
    np.testing.assert_allclose(c.update(observation_time=10.6),[2.,3.],atol=1e-12)

def test_setup_delay_advances_old_command_without_reset(tmp_path):
    import mujoco,time
    from amsrr_det.task_sequence import Sequence
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 3"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);seq=Sequence(tmp_path,1,None)
    seq.data=d;seq.physical=SimpleNamespace(mj=m);seq.air=SimpleNamespace(apply=lambda d:None)
    d.time=12.;seq.pending_wall=time.perf_counter()-.05
    seq.resume(seq.advance_plain)
    assert d.time>=12.05 and d.qpos[2]<3. and d.qvel[2]<0
    assert seq.pauses[0]['from_s']==12.

def test_reordered_actuators_are_matched_by_name():
    import mujoco
    from amsrr_det.task_sequence import Coordinates,BlendedControl
    world='<worldbody><body><joint name="a"/><geom size=".1"/><body pos=".2 0 0"><joint name="b"/><geom size=".1"/></body></body></worldbody>'
    def model(order):return mujoco.MjModel.from_xml_string('<mujoco>'+world+'<actuator>'+''.join(f'<motor name="{n}" joint="{n}"/>' for n in order)+'</actuator></mujoco>')
    mapping=Coordinates(model('ab'),model('ba'))
    class Control:
        last={}
        def update(self,**kwargs):return np.array([7.,8.])
    blend=BlendedControl(Control(),np.array([1.,2.]),mapping)
    np.testing.assert_array_equal(blend.update(observation_time=1.),[1.,2.])
    np.testing.assert_allclose(blend.update(observation_time=2.),[8.,7.])

def test_task_scene_is_added_before_start_without_changing_robot_physics():
    import mujoco
    from amsrr_det.assembly_plan import prepare
    from amsrr_det.mujoco_model import make_plant
    from amsrr_det.task_sequence import add_scene,Coordinates
    edges=[dict(a=0,b=1,port_a='pitch1',port_b='yaw1')]
    plan=prepare(dict(modules=2,edges=edges,initial_seed=10))
    original=plan['plant'].mj
    task=make_plant(2,edges,dict(mass=.5,size=[.2,.2,.2]),meshes=False,floor=True)
    task.mj.opt.integrator=mujoco.mjtIntegrator.mjINT_DISCRETE;task.mj.opt.impratio=10000.
    d=mujoco.MjData(task.mj);oi=task.mj.jnt_qposadr[task.mj.joint('object').id];d.qpos[oi:oi+3]=[2,3,.1]
    physical=add_scene(plan,task,d)
    m=physical.mj
    assert m.opt.integrator==task.mj.opt.integrator and m.opt.impratio==10000.
    reread=mujoco.MjModel.from_xml_string(physical.xml)
    assert reread.opt.integrator==m.opt.integrator and reread.opt.timestep==m.opt.timestep
    new_object=m.jnt_qposadr[m.joint('object').id]
    np.testing.assert_array_equal(plan['initial'][new_object:new_object+3],[2,3,.1])
    for i in range(1,original.nbody):
        j=m.body(original.body(i).name).id
        assert m.body_mass[j]==original.body_mass[i]
        np.testing.assert_array_equal(m.body_inertia[j],original.body_inertia[i])
    assert m.neq==1 and m.nq>task.mj.nq
    q,v=Coordinates(m,task.mj).observe(plan['initial'],np.zeros(m.nv))
    np.testing.assert_array_equal(q[oi:oi+3],[2,3,.1])

def test_cancel_terminates_only_the_owned_process_group(tmp_path):
    import os,subprocess,sys,time
    from amsrr_det.task_server import Jobs
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    try:
        jobs=Jobs(tmp_path);jobs.active='a'*32;jobs.process=child
        out=tmp_path/jobs.active;out.mkdir()
        jobs.cancel(jobs.active)
        assert child.poll() is not None
        assert json.loads((out/'status.json').read_text())['state']=='cancelled'
    finally:
        if child.poll() is None:child.kill();child.wait()

def test_drive_encoding_switch_keeps_effort_and_physical_state():
    import mujoco
    from amsrr_det.task_sequence import Coordinates,switch_drives
    world='<worldbody><body><joint name="a"/><geom size=".1"/></body></worldbody>'
    def model(k):return mujoco.MjModel.from_xml_string('<mujoco>'+world+f'<actuator><position name="a" joint="a" kp="{k}"/></actuator></mujoco>')
    physical=model(100);control=model(200);d=mujoco.MjData(physical)
    d.qpos[0]=.2;d.qvel[0]=.3;d.ctrl[0]=.5;d.time=3.2
    mujoco.mj_forward(physical,d);old=d.actuator_force.copy();before=(d.qpos.copy(),d.qvel.copy(),d.time)
    switch_drives(physical,control,d,Coordinates(physical,control));mujoco.mj_forward(physical,d)
    np.testing.assert_allclose(d.actuator_force,old,atol=1e-12)
    np.testing.assert_array_equal(d.qpos,before[0]);np.testing.assert_array_equal(d.qvel,before[1]);assert d.time==before[2]


def test_integrated_planning_failure_preserves_reason(tmp_path, monkeypatch):
    from amsrr_det.task_execution import execute
    import scripts.run_walking_task as engine
    monkeypatch.setattr(engine, 'run', lambda *a, **k: dict(
        task_success=False, reason='planning_timeout', planning_s=15.1))
    result = execute(make_spec('walking', defaults('walking'), 'integrated'), tmp_path)
    assert result['success'] is False
    assert result['reason'] == 'planning_timeout'
    assert result['assembly'] is None
    assert result['execution_started'] is False


def test_sequence_reference_reuses_storage_without_mutating_physics(tmp_path):
    import mujoco
    from amsrr_det.mujoco_model import make_plant
    from amsrr_det.assembly import make_assembly_plant
    from amsrr_det.task_sequence import Sequence
    reduced=make_plant(1,meshes=False)
    seq=Sequence(tmp_path,1,None)
    seq.physical=make_assembly_plant(1,[],meshes=False)
    d=mujoco.MjData(seq.physical.mj);before=d.qpos.copy()
    def pose(t):
        q=reduced.mj.qpos0.copy();q[0]=t;return q
    ref=seq.reference(SimpleNamespace(pose=pose),reduced)
    workspace=(id(ref.observed),id(ref.kinematics))
    first=ref.pose(0.);second=ref.pose(1.)
    assert second[0]-first[0]==pytest.approx(1.)
    np.testing.assert_array_equal(first,ref.pose(0.))
    np.testing.assert_array_equal(d.qpos,before)
    assert (id(ref.observed),id(ref.kinematics))==workspace
