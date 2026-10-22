from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
from amsrr_det.grasp_task import GraspTask


@pytest.mark.parametrize('angular_speed,finished',[(.03,True),(.11,False)])
def test_release_uses_separate_linear_and_angular_limits(angular_speed,finished):
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;q[meta['object']['q']+2]=.15
    pose=q[meta['object']['q']:].copy()
    trim=dict(thrust=[1.]*len(meta['rotors']),joints=[dict(effort_nm=0.) for a in meta['actuators']])
    plan=SimpleNamespace(plant=SimpleNamespace(pin=model,meta=meta),initial_q=q,closed_q=q,open_q=q,
        source=pose,destination=pose,support_top=0.,shape=ObjectShape('box',[.3,.3,.3],.6),trims=[trim]*3)
    task=GraspTask(plan);task.stage='RELEASE';task.release_start=q.copy()
    task.grasp=SimpleNamespace(geometry=lambda q:(None,None,None,np.array([-.03])),
        preload=np.array([.00015]),last_diagnostics={},
        update=lambda q,v,goal,**kwargs:(q.copy(),np.zeros(model.nv)))
    task.flight.update=lambda *args,**kwargs:(np.zeros(len(meta['actuators'])),np.zeros(len(meta['rotors'])))
    v=np.zeros(model.nv);v[meta['object']['v']+5]=angular_speed
    for step in range(220):
        task.update(step*.01,q,v)
        if task.finished:break
    assert task.finished==finished


@pytest.mark.parametrize('mode,gap,speed,start_transfer',[
    ('resting',.0001,0.,True),('sequential',.0001,0.,False),
    ('resting',.001,0.,False),('resting',.0001,.03,False)])
def test_resting_transfer_requires_sustained_near_table_observation(mode,gap,speed,start_transfer):
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;q[meta['object']['q']+2]=.15+gap
    pose=q[meta['object']['q']:].copy()
    trim=dict(thrust=[1.]*len(meta['rotors']),joints=[dict(effort_nm=0.) for a in meta['actuators']])
    plan=SimpleNamespace(plant=SimpleNamespace(pin=model,meta=meta),initial_q=q,closed_q=q,
        source=pose,destination=pose,support_top=0.,shape=ObjectShape('box',[.3,.3,.3],.6),trims=[trim]*3)
    task=GraspTask(plan,release_load_mode=mode);task.stage='UNLOAD';task.load=1.;task.release_object_goal=pose.copy()
    task.grasp=SimpleNamespace(geometry=lambda q:(None,None,None,np.array([.0001])),
        preload=np.array([.00015]),last_diagnostics={},
        update=lambda q,v,goal,**kwargs:(q.copy(),np.zeros(model.nv)))
    task.flight.update=lambda *args,**kwargs:(np.zeros(len(meta['actuators'])),np.zeros(len(meta['rotors'])))
    v=np.zeros(model.nv);v[meta['object']['v']]=speed
    for step in range(100):task.update(2.+step*.01,q,v)
    assert (task.support_release_start is not None)==start_transfer
    assert (task.load<1.)==start_transfer


@pytest.mark.parametrize('shift,advances',[(.012,True),(.031,False)])
def test_task_release_window_allows_settling_inside_final_tolerance(shift,advances):
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;oq=meta['object']['q'];q[oq+2]=.1501
    pose=q[oq:oq+7].copy()
    trim=dict(thrust=[1.]*len(meta['rotors']),joints=[dict(effort_nm=0.) for a in meta['actuators']])
    plan=SimpleNamespace(plant=SimpleNamespace(pin=model,meta=meta),initial_q=q,closed_q=q,
        source=pose,destination=pose,support_top=0.,shape=ObjectShape('box',[.3,.3,.3],.6),trims=[trim]*3)
    task=GraspTask(plan,release_pose_mode='task');task.stage='UNLOAD';task.load=1.;task.release_object_goal=pose.copy()
    task.grasp=SimpleNamespace(geometry=lambda q:(None,None,None,np.array([.0001])),
        preload=np.array([.00015]),last_diagnostics={},
        update=lambda q,v,goal,**kwargs:(q.copy(),np.zeros(model.nv)))
    task.flight.update=lambda *args,**kwargs:(np.zeros(len(meta['actuators'])),np.zeros(len(meta['rotors'])))
    q=q.copy();q[oq]+=shift
    task.update(.01,q,np.zeros(model.nv))
    assert (task.release_elapsed>0.)==advances
    assert task.load==1.  # Pose acceptance alone is not separation/support proof.


def test_supported_release_starts_motion_deadline_after_planning():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;oq=meta['object']['q'];q[oq+2]=.15
    pose=q[oq:oq+7].copy();v=np.zeros(model.nv)
    trim=dict(thrust=[1.]*len(meta['rotors']),joints=[dict(effort_nm=0.) for a in meta['actuators']])
    plan=SimpleNamespace(plant=SimpleNamespace(pin=model,meta=meta),initial_q=q,closed_q=q,
        source=pose,destination=pose,support_top=0.,shape=ObjectShape('box',[.3,.3,.3],.6),trims=[trim]*3)
    task=GraspTask(plan,release_method='supported');task.stage='RELEASE'
    task.grasp=SimpleNamespace(geometry=lambda q:(None,None,None,np.array([0.])),last_diagnostics={})
    task.flight.update=lambda *a,**kw:(np.zeros(len(meta['actuators'])),np.zeros(len(meta['rotors'])))
    release=SimpleNamespace(opening_timeout=None,opening_reference_complete=False,
        open=lambda q,v:(q.copy(),v.copy(),0.,task.jg.copy(),task.fg.copy()))
    task.supported_release=release
    task.update(16.,q,v)
    assert task.reason is None and not task.finished
    release.opening_timeout=20.
    task.update(19.,q,v)
    assert task.reason is None and not task.finished
    task.update(20.01,q,v)
    assert task.reason=='phase_timeout:RELEASE'


def test_supported_release_pairs_thrust_with_issued_gimbal_commands():
    from amsrr_det.flight import FlightController
    from amsrr_det.release import SupportedRelease
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;q[meta['object']['q']+2]=.15
    flight=FlightController(model,meta,q)
    flight.command[flight.ai]=.025
    flight.command[flight.dock]=.0005
    reference=q.copy();reference[flight.qi[flight.ai]]=-.4
    docks=[a for a in meta['actuators']if a['role']=='dock']
    grasp=SimpleNamespace(model=model,meta=meta,reference=reference,
        qi=np.array([a['q']for a in docks]),vi=np.array([a['v']for a in docks]),
        last_velocity=np.zeros(6+len(docks)),scale=np.r_[np.full(3,.06),np.full(3,.15),np.full(len(docks),.2)])
    previous_q=q.copy();previous_command=flight.command.copy();previous_force=flight.force.copy()
    release=SupportedRelease(None,grasp,flight,q)
    target,velocity,load,bias,force=release.unload()
    np.testing.assert_array_equal(q,previous_q)
    np.testing.assert_array_equal(force,previous_force)
    np.testing.assert_array_equal(target[flight.qi[flight.ai]],previous_command[flight.ai])
    issued=target[flight.qi]+(bias+flight.kd*velocity[flight.vi])/flight.kp
    np.testing.assert_allclose(issued[flight.dock],previous_command[flight.dock],atol=1e-15)
    assert load==1.
