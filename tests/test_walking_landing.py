"""Airborne allocation and audit-clock contracts independent of saved trials."""
import numpy as np
import mujoco
from amsrr_det.walking import WalkSpec
from amsrr_det.walking_plan import plant_for
from amsrr_det.walking_control import GroundControl
from amsrr_det.walking_audit import WalkingAudit


def test_airborne_allocation_never_borrows_future_support():
    plant=plant_for(dict(modules=1,edges=[]));m=plant.mj
    q=m.qpos0.copy();q[2]=1.
    links=['m0/pitch_grasp_mechanism_1','m0/pitch_grasp_mechanism_2']
    # Resolve actual foot links through the shared task definition.
    from amsrr_det.walking import foot_link
    links=[foot_link(0,'pitch1'),foot_link(0,'pitch2')]
    control=GroundControl(plant,q,links);control.drive.observe(q,np.zeros(m.nv))
    feet=control.positions(control.drive.d);feet[:,2]=0.
    command=control.update(q,np.zeros(m.nv),q[:7],np.zeros(6),np.zeros(6),feet,
        np.zeros_like(feet),np.zeros_like(feet),np.zeros(2,bool),flight_mode=True,ground_fraction=0.,observation_time=0.)
    assert np.isfinite(command).all()
    assert control.last['active_feet']==[]
    np.testing.assert_array_equal(control.last['forces'],np.zeros((2,3)))
    np.testing.assert_array_equal(control.lift_bias,0.)
    assert command[control.drive.nj:].sum()>.8*plant.meta['mass']*9.80665


def test_walking_audit_uses_local_time_without_resetting_physics():
    from types import SimpleNamespace
    from amsrr_det.walking import foot_link
    plant=plant_for(dict(modules=1,edges=[]));m=plant.mj;data=mujoco.MjData(m)
    data.qpos[2]=1.;q=data.qpos.copy();spec=WalkSpec(modules=1,speed=.22)
    reference=SimpleNamespace(period=1.,origin=np.zeros(3),pose=lambda t:q.copy())
    audit=WalkingAudit(plant,[foot_link(0,'pitch1')],spec,reference,start_time=11.3)
    data.time=11.3;original=data.qpos.copy();audit.observe(data)
    assert audit.samples[-1]['time_s']==0.
    assert data.time==11.3
    np.testing.assert_array_equal(data.qpos,original)
    data.time=11.4;audit.observe(data)
    assert abs(audit.samples[-1]['time_s']-.1)<1e-12
