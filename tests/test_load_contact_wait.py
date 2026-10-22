"""Near a pad is not yet permission to increase its carrying load."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
from amsrr_det.grasp_task import GraspTask


def task_at_load():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;q[meta['object']['q']+2]=.15
    pose=q[meta['object']['q']:].copy()
    trim=dict(thrust=[1.]*len(meta['rotors']),
              joints=[dict(effort_nm=0.) for _ in meta['actuators']])
    plan=SimpleNamespace(plant=SimpleNamespace(pin=model,meta=meta),initial_q=q,
        closed_q=q,source=pose,destination=pose,support_top=0.,
        shape=ObjectShape('box',[.3,.3,.3],.6),trims=[trim]*3)
    task=GraspTask(plan);task.stage='LOAD'
    gaps=np.array([.0001,-.00001]);updates=[];issued=[]
    def reference(q,v,goal,**kwargs):
        updates.append(goal.copy());return q.copy(),v.copy()
    task.grasp=SimpleNamespace(geometry=lambda q:(None,None,None,-gaps),
        relative_material_speed=lambda q,v:0.,preload=np.array([.00015]*2),
        last_diagnostics={},update=reference)
    def issue(q,v,target,**kwargs):
        issued.append(kwargs['payload_mass'])
        return task.flight.command.copy(),task.flight.force.copy()
    task.flight.update=issue
    return task,q,np.zeros(model.nv),gaps,updates,issued


def test_contact_wait_keeps_feedback_and_restarts_interrupted_readiness():
    task,q,v,gaps,updates,issued=task_at_load()
    for i in range(100):task.update(i*.01,q,v)
    assert len(updates)==len(issued)==100 and max(issued)==0.
    assert task.load_start is None and task.stage=='LOAD'
    gaps[:]=-.00001
    for i in range(100,120):task.update(i*.01,q,v)
    gaps[0]=.0001;task.update(1.2,q,v)
    gaps[:]=-.00001
    for i in range(121,150):task.update(i*.01,q,v)
    assert task.load_start is None
    task.update(1.5,q,v)
    assert task.load_start==1.5 and task.load==0.
    task.update(2.5,q,v)
    assert task.load==.5 and issued[-1]==.3
    assert task.stage_start==0.  # Readiness does not extend the phase deadline.


def test_wait_cannot_extend_the_original_load_deadline():
    task,q,v,*_=task_at_load()
    for i in range(602):task.update(i*.01,q,v)
    assert task.reason=='phase_timeout:LOAD'
    assert task.load==0. and task.load_start is None
