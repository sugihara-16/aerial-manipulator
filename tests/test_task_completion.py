"""Task completion depends on physical delivery, lift and separation."""
from types import SimpleNamespace

import mujoco
import numpy as np

from amsrr_det.task_audit import task_checks


def test_completion_requires_actual_lift_and_delivery(monkeypatch):
    from build import contact_step_control

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object" pos="0 0 1"><freejoint name="object"/>
        <geom name="object" type="sphere" size=".05"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.time = 3.
    plan = SimpleNamespace(destination=np.array([0., 0., 1., 0., 0., 0., 1.]))
    audit = SimpleNamespace(links=[], last_contact_time=1., max_clearance=.15,
                            min_clearance_carried=.12)
    monkeypatch.setattr(contact_step_control, 'pad_gap_jacobians',
                        lambda *args: (np.array([.03]), None))
    plant = SimpleNamespace(mj=model)
    assert all(task_checks(plant, plan, data, audit).values())
    audit.max_clearance = 0.
    assert not task_checks(plant, plan, data, audit)['lifted']
    data.qpos[0] += .1
    assert not task_checks(plant, plan, data, audit)['position']
