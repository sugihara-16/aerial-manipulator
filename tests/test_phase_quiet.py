import numpy as np
from amsrr_det.grasp_task import quiet_object,bounded_pose_window


def test_linear_and_angular_motion_have_separate_limits():
    assert quiet_object(np.array([.001,0,0,0,.03,0]))
    assert not quiet_object(np.array([.025,0,0,0,0,0]))
    assert not quiet_object(np.array([0,0,0,0,.06,0]))


def test_window_rejects_oscillation_even_when_endpoints_match():
    still=np.tile([0.,0.,0.,0.,0.,0.,1.],(31,1))
    assert bounded_pose_window(still)
    assert not bounded_pose_window(still[:30])
    moving=still.copy();moving[15,0]=.002
    assert not bounded_pose_window(moving)
    moving=still.copy();moving[15,3:]=[0,0,np.sin(.01),np.cos(.01)]
    assert not bounded_pose_window(moving)
    still[::2,3:]*=-1
    assert bounded_pose_window(still)
