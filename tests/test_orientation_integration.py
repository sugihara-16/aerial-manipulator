import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from amsrr_det.numerics import advance_world_orientation


@pytest.mark.parametrize('spin',[[0.,0.,0.],[0.,0.,.02],[.03,-.01,.04]])
def test_long_reference_integration_preserves_commanded_world_rotation(spin):
    # Near-90-degree yaw is where the old repeated matrix conversion amplified
    # tiny non-unit roundoff into an unintended reference rotation.
    initial=Rotation.from_euler('xyz',[-.17,.25,np.pi/2]).as_quat()
    q=initial*(1.-7e-7);spin=np.array(spin)
    for _ in range(10000):
        q=advance_world_orientation(q,.01*spin)
        assert abs(np.linalg.norm(q)-1.)<5e-15
    expected=Rotation.from_rotvec(100*spin)*Rotation.from_quat(initial)
    assert (Rotation.from_quat(q)*expected.inv()).magnitude()<1e-11
