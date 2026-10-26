import mujoco
import numpy as np
from build.staged_approach import AcquisitionReference


def test_contact_handoff_has_no_pose_or_gap_jump_and_reaches_target():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    first = model.qpos0.copy(); target = first.copy()
    target[:3] = [.02, -.01, .03]
    target[3:7] = [np.cos(.2), 0., 0., np.sin(.2)]
    gaps, compression = np.array([.01, .03]), np.array([.00005, .0001])
    path = AcquisitionReference(model, first, gaps)
    q, c = path.update(target, compression, .01)
    np.testing.assert_allclose(q, first, atol=1e-14)
    np.testing.assert_allclose(c, -gaps, atol=1e-14)
    next_q, next_c = path.update(target, compression, .01)
    assert np.linalg.norm(next_q-q) < 1e-7
    assert np.linalg.norm(next_c-c) < 1e-7
    for _ in range(410):q,c=path.update(target, compression, .01)
    np.testing.assert_allclose(q, target, atol=1e-12)
    np.testing.assert_allclose(c, compression, atol=1e-12)
