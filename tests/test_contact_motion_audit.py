"""Pressure redistribution is different from displacement of robot material."""
from types import SimpleNamespace
import numpy as np
import mujoco
import pytest
from amsrr_det.task_audit import TaskAudit
from amsrr_det.fast_task_audit import FastTaskAudit


@pytest.mark.parametrize('audit_type',[TaskAudit,FastTaskAudit,lambda *a,**kw:FastTaskAudit(*a,clearance_period=.1,**kw)])
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


def test_sampled_clearance_does_not_skip_physical_actuator_audit():
    # The sampled check is geometry-only. Actuator forces are still inspected
    # at every physics step, including between consecutive clearance queries.
    m=mujoco.MjModel.from_xml_string('''<mujoco><option cone="elliptic"/>
      <worldbody><body name="pad"><joint name="motor"/>
      <geom name="pad/g" type="sphere" size=".01"/></body>
      <body name="object" pos="0 0 1"><freejoint/><geom name="object" type="sphere" size=".1"/></body></worldbody>
      <actuator><motor name="motor" joint="motor"/></actuator></mujoco>''')
    p=SimpleNamespace(mj=m,meta=dict(actuators=[dict(name='motor',lower=-1,upper=1,effort=1)]))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    audit=FastTaskAudit(p,['pad'],[.2,.2,.2],0.,clearance_period=.1)
    audit.update(d);queries=audit.proximity.queries
    d.time=.01;d.actuator_force[0]=1.5;audit.update(d)
    assert audit.proximity.queries==queries
    assert audit.max_effort_ratio==1.5
    assert audit.result()['clearance_sampling_period_s']==.1


@pytest.mark.parametrize('include_wheel',[False,True])
def test_face_audit_detects_top_contact_between_saved_samples(include_wheel):
    wheel='<geom name="valve_wheel" type="sphere" pos="0 .1099 -.0199" size=".01" mass="0" condim="3"/>' if include_wheel else ''
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option cone="elliptic" gravity="0 0 0" timestep=".001"/>
      <worldbody><body name="m0/pad" pos="0 .1099 1"><joint name="motor"/>
      <geom name="m0/pad/g" type="sphere" size=".01" mass=".1" condim="3"/></body>
      <body name="object" pos="0 0 1"><freejoint name="object_joint"/>
      <geom name="object" type="box" size=".1 .1 .1" mass="1" condim="3"/>{wheel}</body></worldbody>
      <actuator><motor name="motor" joint="motor"/></actuator></mujoco>''')
    p=SimpleNamespace(mj=m,meta=dict(actuators=[dict(name='motor',lower=-1,upper=1,effort=1)]))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    audit=FastTaskAudit(p,['m0/pad'],[.2,.2,.2],0.,clearance_period=.1,face_axis=1)
    audit.face_active=True
    assert audit.advance(d,1) is None
    if include_wheel:
        assert not audit.result()['contact_faces']['passes']
        assert audit.result()['contact_faces']['maximum_non_side_normal_force_n']>0.
        return
    assert audit.result()['contact_faces']['passes']
    oq=int(m.jnt_qposadr[m.joint('object_joint').id])
    d.qpos[oq:oq+3]=[0.,.1099,.8901];d.qvel[:]=0.
    assert audit.advance(d,1) is None  # diagnostic violation is not a stop
    d.qpos[oq:oq+3]=[0.,0.,1.];d.qvel[:]=0.
    assert audit.advance(d,1) is None
    result=audit.result()['contact_faces']
    assert result['integration_steps']==3 and result['violating_steps']==1
    assert result['first_violation_s']==pytest.approx(.001)
    assert result['maximum_non_side_normal_force_n']>0.
    assert abs(result['worst_normal_local'][2])>.99
    assert not result['passes']
