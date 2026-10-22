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
