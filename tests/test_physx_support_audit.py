"""Independent evaluator geometry catches penetration without report events."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.objects import ObjectShape
from amsrr_det.mujoco_model import make_plant,CONTACT_CACHE
from build.physx_audit import PhysXAudit


@pytest.mark.parametrize('family',['box','cylinder'])
def test_support_penetration_is_checked_when_physx_reports_no_contacts(family):
 shape=ObjectShape(family,[.3,.3,.3],.6);plant=make_plant(1,[],payload=shape.payload(),meshes=False,floor=True)
 q=pin.neutral(plant.pin);q[2]=1.;q[plant.meta['object']['q']+2]=.15-.0025
 sim=SimpleNamespace(time=.001,interface=SimpleNamespace(subscribe_full_contact_report_events=lambda callback:None))
 audit=PhysXAudit(plant,[],shape,0.,sim,{});command=np.array([q[a['q']] for a in plant.meta['actuators']])
 audit.update(q,np.zeros(plant.pin.nv),command,.001)
 assert audit.reported_points==0 and len(audit.last_contact_rows)==0
 assert abs(audit.max_independent_support_penetration-.0025)<1e-8
 assert audit.max_penetration>.002


@pytest.mark.parametrize('allowed',[False,True])
def test_selected_gripper_contact_is_rejected_before_acquisition(allowed):
 shape=ObjectShape('box',[.3,.3,.3],.6);plant=make_plant(1,[],payload=shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
 q=pin.neutral(plant.pin);q[2]=1.;q[plant.meta['object']['q']+2]=.15
 sim=SimpleNamespace(time=.001,interface=SimpleNamespace(subscribe_full_contact_report_events=lambda callback:None))
 link='m0/yaw_dock_mech1';audit=PhysXAudit(plant,[link],shape,0.,sim,{})
 pad=next(i for i,name in enumerate(audit.names) if name.startswith(link+'/'))
 # A real contact report's 0.1 N normal load must not be exempted merely
 # because this is a pad that will be selected in a later task phase.
 audit.contacts=[(pad,audit.object_geom,0.,0.,.15,0.,0.,.0001,0.,0.,1.,0.)]
 audit.contact_shapes=[(100,200)]
 command=np.array([q[a['q']] for a in plant.meta['actuators']])
 audit.update(q,np.zeros(plant.pin.nv),command,.001,grasp_contact_allowed=allowed)
 assert bool(audit.other_contacts)==(not allowed)
 np.testing.assert_allclose(audit.last_normal_loads,[.1])


@pytest.mark.parametrize('layer,normal,internal',[
 ('layer_0',[0,0,1],False),('layer_1',[0,0,-1],False),
 ('layer_0',[1,0,0],False),('layer_1',[1,0,0],False),
 ('layer_0',[0,0,-1],True),('layer_1',[0,0,1],True),
 ('layer_1',[.6,0,.8],True),
])
def test_partition_audit_distinguishes_outer_faces_and_new_cut_edges(layer,normal,internal):
 shape=ObjectShape('box',[.3,.3,.3],.6);plant=make_plant(1,[],payload=shape.payload(),meshes=False,floor=True)
 sim=SimpleNamespace(time=.001,interface=SimpleNamespace(subscribe_full_contact_report_events=lambda callback:None))
 geometry=dict(lookup={100:plant.mj.geom('object').id,200:plant.mj.geom('floor').id},
  support_layer_ids={200:dict(geom=plant.mj.geom('floor').id,layer=layer,rotation=np.eye(3))})
 for object_first in [True,False]:
  audit=PhysXAudit(plant,[],shape,0.,sim,geometry)
  n=np.asarray(normal)*(1 if object_first else -1)
  vector=lambda a:SimpleNamespace(x=a[0],y=a[1],z=a[2])
  header=SimpleNamespace(num_contact_data=1,collider0=100 if object_first else 200,collider1=200 if object_first else 100,contact_data_offset=0)
  for impulse in [0.,.01]:
   audit._contacts([header],[SimpleNamespace(position=vector([0,0,0]),normal=vector(n),impulse=vector(n*impulse),separation=0.)])
   assert abs(audit.max_internal_support_impulse-(impulse if internal else 0.))<1e-12


@pytest.mark.parametrize('object_first',[True,False])
def test_friction_contributes_to_pad_force_and_moment_but_not_normal_load(object_first):
 shape=ObjectShape('box',[.3,.3,.3],.6);plant=make_plant(1,[],payload=shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
 q=pin.neutral(plant.pin);q[2]=1.;q[plant.meta['object']['q']+2]=.15
 sim=SimpleNamespace(time=.001,interface=SimpleNamespace(subscribe_full_contact_report_events=lambda callback:None))
 link='m0/yaw_dock_mech1';audit=PhysXAudit(plant,[link],shape,0.,sim,{})
 pad=next(i for i,name in enumerate(audit.names) if name.startswith(link+'/'))
 import mujoco
 near=audit.proximity_data;near.qpos[:]=plant.configuration(q);mujoco.mj_kinematics(audit.proximity_model,near)
 origin=near.xpos[audit.link_bodies[link]].copy()
 pair=[audit.object_geom,pad] if object_first else [pad,audit.object_geom];sign=-1 if object_first else 1
 # Normal 25 N alone passes the 30 N bound, but with 20 N of friction
 # the pad resultant is 32 N. Tangential load has a separate moment arm.
 audit.contacts=[[*pair,*origin,0.,0.,sign*.025,0.,0.,float(sign),0.]]
 audit.contact_shapes=[(100,200)]
 audit.friction=[[*pair,*(origin+[0.,.1,0.]),sign*.020,0.,0.]]
 audit.friction_shapes=[(100,200)]
 command=np.array([q[a['q']] for a in plant.meta['actuators']])
 audit.update(q,np.zeros(plant.pin.nv),command,.001)
 np.testing.assert_allclose(audit.last_forces,[[20.,0.,25.]])
 np.testing.assert_allclose(audit.last_moments,[[0.,0.,-2.]])
 np.testing.assert_allclose(audit.last_normal_loads,[25.])
 assert audit.max_force>30. and audit.max_moment==pytest.approx(2.)
 assert not audit.friction and len(audit.last_friction_rows)==1


@pytest.mark.parametrize('object_first',[True,False])
@pytest.mark.parametrize('height,internal',[(1.,False),(.996,True)])
def test_support_cut_friction_is_checked_even_without_a_normal_impulse(object_first,height,internal):
 shape=ObjectShape('box',[.3,.3,.3],.6);plant=make_plant(1,[],payload=shape.payload(),meshes=False,floor=True)
 sim=SimpleNamespace(time=.001,interface=SimpleNamespace(subscribe_full_contact_report_events=lambda callback:None))
 geometry=dict(lookup={100:plant.mj.geom('object').id,200:plant.mj.geom('floor').id},
  support_layer_ids={200:dict(geom=plant.mj.geom('floor').id,layer='layer_0',rotation=np.eye(3),center=np.array([0.,0.,.5]),half_size=np.array([1.,1.,.5]))})
 audit=PhysXAudit(plant,[],shape,0.,sim,geometry)
 vector=lambda x:SimpleNamespace(x=x[0],y=x[1],z=x[2])
 header=SimpleNamespace(num_contact_data=0,collider0=100 if object_first else 200,collider1=200 if object_first else 100,contact_data_offset=0,num_friction_anchors_data=1,friction_anchors_offset=0)
 audit._contacts([header],[],[SimpleNamespace(position=vector([0.,0.,height]),impulse=vector([.01,0.,0.]))])
 assert audit.max_internal_support_friction_impulse==pytest.approx(.01 if internal else 0.)
 assert audit.max_internal_support_impulse==pytest.approx(.01 if internal else 0.)
