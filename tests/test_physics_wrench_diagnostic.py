import numpy as np
from scipy.spatial.transform import Rotation
from build.physics_diagnostics import world_wrench


def test_force_diagnostic_uses_body_frame_and_common_fixed_origin():
    poses=np.array([[1.,2.,3.,*Rotation.from_euler('y',90,degrees=True).as_quat()],
                    [-1.,0.,1.,0.,0.,0.,1.]])
    forces=np.array([[0.,0.,2.],[0.,3.,0.]])
    moments=np.array([[0.,1.,0.],[1.,0.,0.]])
    expected=np.r_[[2.,3.,0.],np.array([1.,1.,0.])+np.cross([1.,2.,3.],[2.,0.,0.])+np.cross([-1.,0.,1.],[0.,3.,0.])]
    np.testing.assert_allclose(world_wrench(poses,forces,moments,np.zeros(3)),expected,atol=1e-12)
    shift=np.array([.4,-.2,.7]);moved=world_wrench(poses,forces,moments,shift)
    np.testing.assert_allclose(moved[:3],expected[:3],atol=1e-12)
    np.testing.assert_allclose(moved[3:],expected[3:]-np.cross(shift,expected[:3]),atol=1e-12)
