"""Pressure redistribution is different from displacement of robot material."""
from types import SimpleNamespace
import numpy as np
import mujoco
import pytest
from amsrr_det.task_audit import TaskAudit
from amsrr_det.fast_task_audit import FastTaskAudit


@pytest.mark.parametrize('audit_type',[TaskAudit,FastTaskAudit])
def test_pressure_center_shift_does_not_count_as_slip_but_material_motion_does(audit_type):
    m=mujoco.MjModel.from_xml_string('''<mujoco><option cone="elliptic"/>
    <worldbody><body name="pad" pos="0 0 .4"><joint name="pad_joint" type="hinge" axis="0 0 1"/>
    <geom name="pad/left" type="sphere" size=".01" pos="-.05 0 0" mass=".1" condim="3"/>
    <geom name="pad/right" type="sphere" size=".01" pos=".05 0 0" mass=".1" condim="3"/></body>
    <body name="object" pos="0 0 .4599"><freejoint name="object_joint"/>
    <geom name="object" type="box" size=".1 .1 .05" mass="1" condim="3"/></body></worldbody>
    <actuator><motor name="pad_joint" joint="pad_joint"/></actuator></mujoco>''')
    p=SimpleNamespace(mj=m,meta=dict(actuators=[dict(name='pad_joint',lower=-1,upper=1,effort=1)]))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    audit=audit_type(p,['pad'],[.2,.2,.1],0.)
    assert d.ncon==2
    rows=np.array([c.efc_address for c in d.contact])
    d.efc_force[rows]=[9.,1.];audit.update(d,carried=True)
    d.efc_force[rows]=[1.,9.];audit.update(d,carried=True)
    assert audit.result()['contact_center_displacement_m']>.03
    assert audit.result()['contact_material_displacement_m']<1e-12
    d.qpos[m.jnt_qposadr[m.joint('object_joint').id]]+=.01
    mujoco.mj_forward(m,d);d.efc_force[np.array([c.efc_address for c in d.contact])]=5.
    audit.update(d,carried=True)
    np.testing.assert_allclose(audit.result()['contact_material_displacement_m'],.01,atol=1e-10)
