import numpy as np
import pinocchio as pin
from amsrr_det.objects import ObjectShape
from amsrr_det.evaluation_inputs import resting_pose


def test_stable_box_face_is_kept_but_unstable_ellipsoid_axis_is_not():
    seed=np.r_[np.zeros(3),pin.Quaternion(pin.rpy.rpyToMatrix(np.pi/2,0,.7)).coeffs()]
    box=ObjectShape('box',[.252,.36,.198],.8)
    pose,report=resting_pose(box,seed)
    np.testing.assert_allclose(pin.Quaternion(pose[3:]).matrix(),pin.Quaternion(seed[3:]).matrix(),atol=1e-10)
    ellipse=ObjectShape('ellipsoid',box.size,.8);pose,report=resting_pose(ellipse,seed)
    assert abs(pose[2]-.099)<1e-8
    R=pin.Quaternion(pose[3:]).matrix()
    for a in np.linspace(0,2*np.pi,16,endpoint=False):
        perturbed=pin.exp3(.001*np.array([np.cos(a),np.sin(a),0.]))@R
        assert -ellipse.lowest_z(np.r_[np.zeros(3),pin.Quaternion(perturbed).coeffs()])>=pose[2]-1e-10
