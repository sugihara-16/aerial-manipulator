import json
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.assembly_air import interference,AirParameters
from amsrr_det.assembly_plan import initial_xy,groups,prepare
from amsrr_det.assembly_fsm import AssemblyFSM


def test_initial_spacing_is_per_module_not_just_one_close_pair():
    for n in range(2,9):
        for seed in range(12):
            xy,yaw=initial_xy(n,seed)
            distances=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
            np.fill_diagonal(distances,np.inf)
            assert np.all((distances.min(axis=1)>=1.)&(distances.min(axis=1)<=2.))
            assert np.all(abs(yaw)<=np.pi)


def test_wake_disappears_with_source_off_or_same_vehicle():
    xyz=np.array([[0.,0.,1.],[0.,0.,.5]])
    axes=np.tile([0.,0.,1.],(2,1));phase=np.zeros((2,2));p=AirParameters()
    loss,_=interference(xyz,axes,np.array([5.,5.]),np.array([0,1]),phase,0.,p)
    assert loss[1]>.1 and loss[1]>loss[0]
    loss,_=interference(xyz,axes,np.array([0.,5.]),np.array([0,1]),phase,0.,p)
    assert loss[1]==0.
    loss,_=interference(xyz,axes,np.array([5.,5.]),np.array([0,0]),phase,0.,p)
    assert np.all(loss==0.)


def test_wake_rotates_with_rotors_and_is_not_fixed_world_down():
    xyz=np.array([[0.,0.,1.],[.04,0.,.5]])
    axes=np.tile([0.,0.,1.],(2,1));phase=np.zeros((2,2));force=np.array([5.,4.]);modules=np.array([0,1]);p=AirParameters()
    R=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    a=interference(xyz,axes,force,modules,phase,.3,p)
    b=interference(xyz@R.T,axes@R.T,force,modules,phase,.3,p)
    np.testing.assert_allclose(a,b)


def test_parallel_rounds_do_not_share_connected_components():
    edges=[dict(a=0,b=1,port_a='pitch1',port_b='yaw1'),dict(a=1,b=2,port_a='pitch1',port_b='yaw1')]
    assert len(set(groups(3,edges,0)))==3
    assert groups(3,edges,1)[0]==groups(3,edges,1)[1]
    assert len(set(groups(3,edges,3)))==1


def test_timeout_cannot_become_success_at_verification():
    case=dict(modules=1,edges=[],initial_seed=10)
    plan=prepare(case);fsm=AssemblyFSM(plan);data=mujoco.MjData(plan['plant'].mj)
    data.qpos[:]=plan['initial'];fsm.stage='VERIFY';fsm.clock=1.;fsm.dwell=.49;data.time=31.
    fsm.update(data)
    assert fsm.failure=='assembly_timeout' and not fsm.success


def test_panel_keeps_failed_downstream_tasks_and_all_plans():
    from scripts.make_assembly_panel import panel
    result=panel()
    assert result['counts']==dict(grasp=149,valve=35,walking=35)
    assert len(result['cases'])==219 and result['module_counts'][1]==8
    assert any(c['id']=='grasp/box_y1_1' for c in result['cases'])


def test_wake_is_continuous_across_source_rotor_plane():
    axes=np.tile([0.,0.,1.],(2,1));phase=np.zeros((2,2));force=np.array([5.,5.]);modules=np.array([0,1])
    values=[]
    for z in [-1e-8,1e-8]:
        loss,_=interference(np.array([[0.,0.,0.],[.12,0.,z]]),axes,force,modules,phase,0.,AirParameters())
        values.append(loss)
    np.testing.assert_allclose(values[0],values[1],rtol=1e-5)


def test_role_assignment_preserves_sampled_physical_positions_and_headings():
    case=dict(modules=3,initial_seed=912,edges=[dict(a=0,b=1,port_a='pitch1',port_b='yaw1'),dict(a=0,b=2,port_a='pitch2',port_b='yaw1')])
    plan=prepare(case);xy,yaw=initial_xy(3,912);assignment=plan['evidence']['physical_module_assignment']
    assert sorted(assignment)==list(range(3))
    for role,physical in enumerate(assignment):
        k=plan['qi'][role];q=plan['initial'][k:k+7]
        np.testing.assert_allclose(q[:2],xy[physical])
        np.testing.assert_allclose(q[3:],np.array([np.cos(yaw[physical]/2),0.,0.,np.sin(yaw[physical]/2)]))
