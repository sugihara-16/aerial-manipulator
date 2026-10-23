import numpy as np
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


def test_symbolic_solver_runs_through_production_entry_and_rechecks_wrench():
    plant=dock_mit_model(make_plant(2,payload={'shape':'box','size':[.3,.2,.2],'mass':.7},mesh_cache=CONTACT_CACHE),kp=200.)
    q=plant.configuration(pin.neutral(plant.pin));local=LocalGrasp(plant,q,[])
    design=GimbalWrenchDesign(local,q,[0.,0.,0.],np.zeros((1,6)))
    thrust=np.full(len(local.sites),plant.mj.body_mass.sum()*9.80665/len(local.sites))
    pose,info=design.solve(np.r_[local.kp*q[local.qi],thrust])
    assert info['accepted']
    assert info['external_wrench_residual']<1e-6
    np.testing.assert_allclose(pose,q,atol=1e-8)


def test_cached_program_reuses_equations_but_not_previous_physics():
    from build.task_gimbal_design import _gimbal_solver
    _gimbal_solver.cache_clear()
    for mass,displacement in [(.4,[.2,0,0]),(1.2,[-.1,.3,0])]:
        plant=dock_mit_model(make_plant(2,payload={'shape':'box','size':[.3,.2,.2],'mass':mass},mesh_cache=CONTACT_CACHE),kp=200.)
        q=plant.configuration(pin.neutral(plant.pin));local=LocalGrasp(plant,q,[])
        corrections=np.zeros((2,6));corrections[1,2]=mass*.2
        design=GimbalWrenchDesign(local,q,displacement,corrections)
        thrust=np.full(len(local.sites),plant.mj.body_mass.sum()*9.80665/len(local.sites))
        _,info=design.solve(np.r_[local.kp*q[local.qi],thrust])
        assert info['accepted']
        assert info['external_wrench_residual']<1e-6
    assert _gimbal_solver.cache_info().hits==1
