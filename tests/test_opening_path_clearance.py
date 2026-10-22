"""A collision between the old five samples must not reach execution."""
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.mujoco_model import make_plant
from amsrr_det.objects import ObjectShape
from amsrr_det.release import check_opening_clearance


def small_scene(monkeypatch,side_offset):
    shape=ObjectShape('box',[.01,.01,.01],.1)
    plant=make_plant(1,payload=shape.payload(),meshes=False)
    doc=ET.fromstring(plant.xml)
    body=doc.find(".//body[@name='m0/main_body']")
    ET.SubElement(body,'geom',name='m0/main_body/probe',type='box',size='.005 .005 .005')
    plant.xml=ET.tostring(doc,encoding='unicode');plant.mj=mujoco.MjModel.from_xml_string(plant.xml)
    first=pin.neutral(plant.pin);first[2]=1.
    data=plant.pin.createData();pin.framesForwardKinematics(plant.pin,data,first)
    oq=plant.meta['object']['q'];first[oq:oq+3]=data.oMf[plant.meta['frames']['m0/main_body']].translation+[.06,side_offset,0.]
    last=first.copy();last[0]+=.16
    monkeypatch.setattr('amsrr_det.mujoco_model.make_plant',lambda *args,**kwargs:copy.copy(plant))
    return plant,shape,first,last


def test_opening_rejects_collision_between_old_samples(monkeypatch):
    plant,shape,first,last=small_scene(monkeypatch,0.)
    m=plant.mj;data=mujoco.MjData(m);ids=[m.geom(name).id for name in ['m0/main_body/probe','object']]
    for fraction in np.linspace(0,1,5):
        data.qpos[:]=plant.configuration(pin.interpolate(plant.pin,first,last,float(fraction)))
        mujoco.mj_kinematics(m,data)
        assert mujoco.mj_geomDistance(m,data,*ids,1.,None)>.001
    with pytest.raises(ValueError,match='clearance_violation') as caught:
        check_opening_clearance(plant.meta,shape,[],[first,last],[])
    assert .25<caught.value.evidence['fraction']<.5
    assert caught.value.evidence['distance_m']<.001


def test_opening_certifies_clear_interpolation(monkeypatch):
    plant,shape,first,last=small_scene(monkeypatch,.05)
    result=check_opening_clearance(plant.meta,shape,[],[first,last],[])
    assert result['intervals']>=1


@pytest.mark.parametrize('offset,clear',[(0.,False),(.0105,False),(.010998,False),(.011002,True),(.05,True)])
def test_swept_boxes_respect_clearance_boundary(monkeypatch,offset,clear):
    # Both boxes have 5 mm half-width. The required gap is 1 mm, with the
    # existing 1 micrometre numerical comparison tolerance.
    plant,shape,first,last=small_scene(monkeypatch,offset)
    if clear:check_opening_clearance(plant.meta,shape,[],[first,last],[])
    else:
        with pytest.raises(ValueError,match='clearance_violation'):
            check_opening_clearance(plant.meta,shape,[],[first,last],[])


def test_shared_base_motion_does_not_hide_relative_joint_collision(monkeypatch):
    shape=ObjectShape('box',[.01,.01,.01],.1)
    plant=make_plant(1,payload=shape.payload(),meshes=False)
    first=pin.neutral(plant.pin);first[2]=1.
    oq=plant.meta['object']['q'];first[oq:oq+3]=[5.,0.,1.]
    joint=next(a for a in plant.meta['actuators'] if a['name']=='m0/yaw_dock_mech_joint1')
    first[joint['q']]=-.2
    last=first.copy();last[joint['q']]=.2
    # Both parts share a translated and rotated base. Only one hinge changes
    # their relative geometry, and it collides between the old five samples.
    last[:3]+=[.1,.05,.02]
    last[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.1,.2,.4)).coeffs()
    mid=pin.interpolate(plant.pin,first,last,.375)
    data=plant.pin.createData();pin.framesForwardKinematics(plant.pin,data,mid)
    names=['m0/yaw_dock_mech1','m0/yaw_dock_mech2'];point=np.array([.07,.01,0.])
    center=data.oMf[plant.meta['frames'][names[0]]].act(point)
    other=data.oMf[plant.meta['frames'][names[1]]].actInv(center)
    doc=ET.fromstring(plant.xml)
    for name,pos in zip(names,[point,other]):
        body=doc.find(f".//body[@name='{name}']")
        ET.SubElement(body,'geom',name=name+'/probe',type='sphere',size='.0005',pos=' '.join(map(str,pos)))
    plant.xml=ET.tostring(doc,encoding='unicode');plant.mj=mujoco.MjModel.from_xml_string(plant.xml)
    monkeypatch.setattr('amsrr_det.mujoco_model.make_plant',lambda *args,**kwargs:copy.copy(plant))
    m=plant.mj;d=mujoco.MjData(m);ids=[m.geom(name+'/probe').id for name in names]
    for u in np.linspace(0,1,5):
        d.qpos[:]=plant.configuration(pin.interpolate(plant.pin,first,last,float(u)))
        mujoco.mj_kinematics(m,d)
        assert mujoco.mj_geomDistance(m,d,*ids,1.,None)>.001
    with pytest.raises(ValueError,match='clearance_violation') as caught:
        check_opening_clearance(plant.meta,shape,[],[first,last],[])
    assert .25<caught.value.evidence['fraction']<.5


def test_reusing_full_candidate_geometry_preserves_path_certificate(monkeypatch):
    plant,shape,first,last=small_scene(monkeypatch,.05)
    fresh=check_opening_clearance(plant.meta,shape,[],[first,last],[])
    reused=check_opening_clearance(plant.meta,shape,[],[first,last],[],prepared_plant=plant)
    assert fresh==reused
    # A colliding path must still be rejected when no model is rebuilt.
    plant,shape,first,last=small_scene(monkeypatch,0.)
    with pytest.raises(ValueError,match='clearance_violation'):
        check_opening_clearance(plant.meta,shape,[],[first,last],[],prepared_plant=plant)
