from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from amsrr_det.free_flight_switch import FreeFlightSwitch
from amsrr_det.mujoco_model import make_plant
from amsrr_det.objects import ObjectShape


def controller(value):
    return SimpleNamespace(command=np.array([value]), force=np.array([2*value]),
        last_joint_bias=np.array([value/2]), last_target_velocity=np.array([value/3]),
        control_count=10)


def test_grasp_loss_during_carry_never_enables_contact_free_control():
    switch=FreeFlightSwitch.__new__(FreeFlightSwitch)
    switch.free=controller(1.);switch.contact=controller(2.)
    switch.active=switch.contact;switch.acquired=True;switch.released=False;switch.transitions=[]
    distant=SimpleNamespace(geometry=lambda q:(None,None,None,np.array([-.1,-.1])))
    assert switch.select('CARRY',distant,None,1.) is switch.contact
    assert not switch.released
    selected=switch.select('RELEASE',distant,None,2.)
    assert selected is switch.free
    np.testing.assert_array_equal(selected.command,switch.contact.command)
    np.testing.assert_array_equal(selected.force,switch.contact.force)
    selected.command[0]=8.
    assert switch.contact.command[0]==2.  # No mutable input alias across controllers.
    near=SimpleNamespace(geometry=lambda q:(None,None,None,np.array([0.,-.1])))
    assert switch.select('RELEASE',near,None,3.) is switch.free
    # Reapproach is checked by the geometry forecast, never a silent mode bounce.
    assert len(switch.transitions)==1


def test_forecast_uses_observed_object_motion_and_rejects_an_intermediate_collision():
    plant=make_plant(1,[],payload=ObjectShape('box',[.2,.2,.2],.5).payload(),
                     meshes=False,floor=False)
    model=plant.pin;obj=plant.meta['object'];q=pin.neutral(model);q[2]=1.
    q[obj['q']:obj['q']+3]=[1.,0.,1.];v=np.zeros(model.nv);v[obj['v']]=-.2
    samples=[]
    for _ in range(10):
        guessed=q.copy();guessed[obj['q']+2]-=.1  # Unconstrained fall is irrelevant here.
        samples.append(dict(q=guessed,v=v.copy()))
    switch=FreeFlightSwitch.__new__(FreeFlightSwitch);switch.plant=plant
    switch.free=SimpleNamespace(prediction_samples=samples,dt=.01);switch.active=switch.free
    switch.last_forecast=None;positions=[]
    joint=plant.mj.joint(model.names[obj['joint']]);adr=joint.qposadr[0]
    def check(qpos,grasp_contact_allowed):
        assert not grasp_contact_allowed
        positions.append(qpos[adr:adr+3].copy())
        hit=len(positions)==4
        return ([('pad','object')] if hit else []),(.0008 if hit else .003),.04
    switch.clearance=SimpleNamespace(check=check)
    assert not switch.check(q,v)
    np.testing.assert_allclose(positions,[[1.-.0002*i,0.,1.]for i in range(1,5)],atol=1e-12)
    assert switch.last_forecast['prediction_time_s']==.004
    assert q[obj['q']+2]==1.
