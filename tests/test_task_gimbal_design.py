import numpy as np
import pytest
import pinocchio as pin
from amsrr_det.mujoco_model import make_plant,CONTACT_CACHE
from build.mit_drive import dock_mit_model
from build.sticking_design import LocalGrasp
from build.momentum_feedback import rigid_motion_map
from build.task_gimbal_design import GimbalWrenchDesign


def test_geometric_wrench_matches_generalized_inputs_and_independent_derivative():
    plant=dock_mit_model(make_plant(2,payload={'shape':'box','size':[.3,.2,.2],'mass':.7},mesh_cache=CONTACT_CACHE),kp=200.)
    q=plant.configuration(pin.neutral(plant.pin));local=LocalGrasp(plant,q,[])
    design=GimbalWrenchDesign(local,q,[.3,-.2,0.]);angles=np.linspace(-.2,.2,design.ng)
    A,b,da,db=design.geometry(angles)
    pose=q.copy();pose[design.qi]=angles;local.at(pose)
    mass,B,_=local.matrices();T=rigid_motion_map(plant.mj,pose)
    np.testing.assert_allclose(A,(T.T@B)[:,local.nj:],atol=2e-13)
    expected=np.array([T.T@(local.d.qfrc_bias+mass@T@np.r_[a,np.zeros(3)])for a in design.accelerations])
    np.testing.assert_allclose(b,expected,atol=2e-12)
    for i in range(design.ng):
        shift=np.zeros(design.ng);shift[i]=1e-6
        ap,bp,_,_=design.geometry(angles+shift);am,bm,_,_=design.geometry(angles-shift)
        np.testing.assert_allclose(da[i],(ap-am)/2e-6,atol=5e-9)
        np.testing.assert_allclose(db[i],(bp-bm)/2e-6,atol=5e-8)


@pytest.mark.parametrize('minimize_input',[False,True])
def test_symbolic_solver_runs_through_production_entry_and_rechecks_wrench(minimize_input):
    plant=dock_mit_model(make_plant(2,payload={'shape':'box','size':[.3,.2,.2],'mass':.7},mesh_cache=CONTACT_CACHE),kp=200.)
    q=plant.configuration(pin.neutral(plant.pin));local=LocalGrasp(plant,q,[])
    design=GimbalWrenchDesign(local,q,[0.,0.,0.],np.zeros((1,6)))
    thrust=np.full(len(local.sites),plant.mj.body_mass.sum()*9.80665/len(local.sites))
    pose,info=design.solve(np.r_[local.kp*q[local.qi],thrust],minimize_input=minimize_input)
    assert info['accepted']
    assert info['external_wrench_residual']<1e-6
    if minimize_input:
        # Jensen is a lower bound; torque balance may require unequal thrust.
        lower=(plant.mj.body_mass.sum()*9.80665)**1.5/np.sqrt(len(local.sites))
        _,nearest=design.solve(np.r_[local.kp*q[local.qi],thrust])
        assert lower-1e-7<=info['mean_scenario_thrust_burden']<=nearest['mean_scenario_thrust_burden']+1e-7
        unchanged=np.ones(len(q),dtype=bool);unchanged[design.qi]=False
        np.testing.assert_array_equal(pose[unchanged],q[unchanged])
    else:
        np.testing.assert_allclose(pose,q,atol=1e-8)


def test_cached_program_reuses_equations_but_not_previous_physics():
    from build.task_gimbal_design import _gimbal_solver
    _gimbal_solver.cache_clear()
    for mass,displacement,count in [(.4,[.2,0,0],2),(1.2,[-.1,.3,0],5)]:
        plant=dock_mit_model(make_plant(2,payload={'shape':'box','size':[.3,.2,.2],'mass':mass},mesh_cache=CONTACT_CACHE),kp=200.)
        q=plant.configuration(pin.neutral(plant.pin));local=LocalGrasp(plant,q,[])
        corrections=np.zeros((count,6));corrections[1,2]=mass*.2
        design=GimbalWrenchDesign(local,q,displacement,corrections)
        thrust=np.full(len(local.sites),plant.mj.body_mass.sum()*9.80665/len(local.sites))
        _,info=design.solve(np.r_[local.kp*q[local.qi],thrust])
        assert info['accepted']
        assert info['external_wrench_residual']<1e-6
    assert _gimbal_solver.cache_info().hits==1


def test_joint_traversal_permutation_reuses_program_and_preserves_pose():
    import copy
    from build.task_gimbal_design import _gimbal_solver
    plant=dock_mit_model(make_plant(2,payload={'shape':'box','size':[.3,.2,.2],'mass':.7},mesh_cache=CONTACT_CACHE),kp=200.)
    q=plant.configuration(pin.neutral(plant.pin));local=LocalGrasp(plant,q,[])
    permuted=copy.copy(local);order=np.arange(local.nj)[::-1]
    for field in ('joint_ids','qi','vi','kp','kd','effort','dock','gain'):
        setattr(permuted,field,getattr(local,field)[order])
    _gimbal_solver.cache_clear();poses=[]
    for model in (local,permuted):
        design=GimbalWrenchDesign(model,q,[0.,0.,0.],np.zeros((1,6)))
        thrust=np.full(len(model.sites),plant.mj.body_mass.sum()*9.80665/len(model.sites))
        pose,info=design.solve(np.r_[model.kp*q[model.qi],thrust])
        assert info['accepted'] and info['external_wrench_residual']<1e-6
        poses.append(pose)
    np.testing.assert_allclose(*poses,atol=1e-12)
    assert _gimbal_solver.cache_info().hits==1
