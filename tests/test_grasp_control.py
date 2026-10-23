import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.grasp_control import surface_points
from amsrr_det.grasp_control import GraspReferences
from amsrr_det.grasp_geometry import grasp_frame


@pytest.mark.parametrize('port',['yaw1','yaw2','pitch1','pitch2'])
def test_material_points_do_not_change_with_joint_configuration(port):
    model,meta=build_model();q=pin.neutral(model)
    first,points=surface_points(model,meta,q,(0,port))
    changed=q.copy();changed[:3]=[1.2,-.7,.4]
    changed[3:7]=pin.Quaternion(pin.exp3(np.array([.1,.2,.5]))).coeffs()
    for actuator in meta['actuators']:changed[actuator['q']]=.3
    second,moved=surface_points(model,meta,changed,(0,port))
    assert first==second
    np.testing.assert_array_equal(points,moved)
    assert len(points)>=3 and np.linalg.matrix_rank(points-points[0],tol=1e-6)>=2


@pytest.mark.parametrize('port',['yaw1','pitch2'])
def test_reused_material_vertices_match_direct_source_mesh(port):
    import xml.etree.ElementTree as ET
    import trimesh
    from amsrr_det.model import ROOT,transform
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw2')]
    model,meta=build_model(2,edges);q=pin.neutral(model)
    surface=(1,port);frame,points=surface_points(model,meta,q,surface)
    source=ROOT/'assets/robots/holon/holon.urdf'
    link=('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]
    collision=ET.parse(source).find(f"link[@name='{link}']/collision")
    mesh=collision.find('geometry/mesh')
    vertices=np.asarray(trimesh.load_mesh(source.parent/mesh.get('filename')).vertices)
    vertices=vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    body=meta['frames']['m1/'+link]
    T=model.frames[frame].placement.inverse()*model.frames[body].placement*transform(collision.find('origin'))
    vertices=vertices@T.rotation.T+T.translation
    front=vertices[vertices[:,0]>vertices[:,0].max()-.0001]
    angles=np.linspace(0,2*np.pi,8,endpoint=False)
    indices=np.unique([np.argmax(front[:,1]*np.cos(a)+front[:,2]*np.sin(a)) for a in angles])
    np.testing.assert_array_equal(points,front[indices])
    points[:]=0.
    np.testing.assert_array_equal(surface_points(model,meta,q,surface)[1],front[indices])


def test_joint_at_reference_bound_does_not_freeze_requested_base_motion():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    actuator=next(a for a in meta['actuators'] if a['role']=='dock' and 'yaw' in a['name'])
    q[actuator['q']]=actuator['upper']-.025
    surfaces=[(0,'yaw1'),(0,'yaw2')];data=model.createData()
    pin.framesForwardKinematics(model,data,q);obj=data.oMf[meta['object']['frame']]
    frames=[meta['frames'][grasp_frame(*s)] for s in surfaces]
    points=np.array([obj.actInv(data.oMf[f].translation) for f in frames])
    normals=np.array([obj.rotation.T@data.oMf[f].rotation[:,0] for f in frames])
    controller=GraspReferences(model,meta,q,q,surfaces,points,normals,
                               samples=[(f,np.zeros((1,3))) for f in frames])
    refresh=controller.refresh_counter
    first=controller.geometry(q);second=controller.geometry(q.copy())
    assert controller.refresh_counter==refresh
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    goal=q[meta['object']['q']:].copy();goal[0]+=.02
    reference,_=controller.update(q,np.zeros(model.nv),goal)
    assert reference[0]>q[0]
    assert reference[actuator['q']]<=actuator['upper']-.025+1e-12
    assert len(controller.last_diagnostics['object_error'])==6


def test_release_can_retreat_from_a_single_surface_without_object_motion():
    # With one surface the retreat is entirely a rigid motion. A sticking
    # constraint would remove all of it although the object is already placed.
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    surface=(0,'pitch1');frame=meta['frames'][grasp_frame(*surface)];obj=data.oMf[meta['object']['frame']]
    point=obj.actInv(data.oMf[frame].translation);normal=obj.rotation.T@data.oMf[frame].rotation[:,0]
    controller=GraspReferences(model,meta,q,q,[surface],[point],[normal],samples=[(frame,np.zeros((1,3)))])
    reference,velocity=controller.update(q,np.zeros(model.nv),q[meta['object']['q']:],preload=[-.001],object_feedback=0.,releasing=True)
    pin.forwardKinematics(model,data,q,velocity);pin.updateFramePlacements(model,data)
    pad_velocity=pin.getFrameVelocity(model,data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    assert pad_velocity@(obj.rotation@normal)<-1e-5
    assert np.linalg.norm(velocity[meta['object']['v']:])==0.


def test_acquisition_checks_relative_motion_not_just_stationary_object():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(0,0,np.pi/2)).coeffs()
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    frame=meta['frames'][grasp_frame(0,'yaw1')];obj=data.oMf[meta['object']['frame']]
    controller=GraspReferences(model,meta,q,q,[(0,'yaw1')],[obj.actInv(data.oMf[frame].translation)],[[1,0,0]],samples=[(frame,np.zeros((1,3)))])
    v=np.zeros(model.nv);v[0]=.03
    assert abs(controller.relative_material_speed(q,v)-.03)<1e-10
    v[meta['object']['v']+1]=.03
    assert controller.relative_material_speed(q,v)<1e-10


def test_release_checks_three_pad_directions_and_rejects_unreachable_object_speed():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q);obj=data.oMf[meta['object']['frame']]
    surfaces=[(0,'yaw1'),(0,'yaw2'),(0,'pitch1')];frames=[meta['frames'][grasp_frame(*s)] for s in surfaces]
    points=[obj.actInv(data.oMf[f].translation) for f in frames];normals=[obj.rotation.T@data.oMf[f].rotation[:,0] for f in frames]
    controller=GraspReferences(model,meta,q,q,surfaces,points,normals,samples=[(f,np.zeros((1,3))) for f in frames])
    ov=meta['object']['v'];v=np.zeros(model.nv);v[ov:ov+3]=[-.0002,.0001,0.]
    _,velocity=controller.update(q,v,q[meta['object']['q']:],preload=[-.002]*3,object_feedback=0.,releasing=True)
    assert controller.last_diagnostics['release_direction_feasible']
    pin.forwardKinematics(model,data,q,velocity);pin.updateFramePlacements(model,data)
    for f,n in zip(frames,normals):
        speed=pin.getFrameVelocity(model,data,f,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        assert -n@(speed-v[ov:ov+3])>=-1e-8
    v[ov:ov+3]=-np.asarray(normals[0])
    before=controller.reference.copy()
    reference,_=controller.update(q,v,q[meta['object']['q']:],preload=[-.002]*3,object_feedback=0.,releasing=True)
    assert not controller.last_diagnostics['release_direction_feasible']
    np.testing.assert_array_equal(reference,before)


@pytest.mark.parametrize('motion',[[.0003,0,0,0,0,0],[0,0,0,.0002,-.0001,.0003]])
def test_rigid_transport_does_not_accumulate_grasp_joint_deformation(motion):
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[:3]=[.1,-.2,1.];q[meta['object']['q']:meta['object']['q']+3]=[.3,.1,1.]
    data=model.createData();pin.framesForwardKinematics(model,data,q);obj=data.oMf[meta['object']['frame']]
    surfaces=[(0,'yaw1'),(0,'yaw2'),(0,'pitch1')];frames=[meta['frames'][grasp_frame(*s)] for s in surfaces]
    points=[obj.actInv(data.oMf[f].translation) for f in frames];normals=[obj.rotation.T@data.oMf[f].rotation[:,0] for f in frames]
    refs=GraspReferences(model,meta,q,q,surfaces,points,normals,
        samples=[(f,np.zeros((1,3))) for f in frames],posture_tracking=True)
    twist=np.array(motion);v=np.zeros(model.nv);ov=meta['object']['v'];v[ov:ov+6]=twist
    _,velocity=refs.update(q,v,q[meta['object']['q']:],desired_twist=twist)
    np.testing.assert_allclose(velocity[refs.vi],0,atol=1e-8)
    # Independent forward kinematics: every pad follows the same rigid object
    # twist, including the moment arm between the base and object origins.
    pin.forwardKinematics(model,data,q,velocity);pin.updateFramePlacements(model,data)
    for frame in frames:
        expected=twist[:3]+np.cross(twist[3:],data.oMf[frame].translation-obj.translation)
        actual=pin.getFrameVelocity(model,data,frame,pin.LOCAL_WORLD_ALIGNED).linear
        np.testing.assert_allclose(actual,expected,atol=1e-8)


def test_posture_preference_restores_base_rotation_about_two_contact_points():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[:3]=[.1,-.2,1.];oq=meta['object']['q'];q[oq:oq+3]=[.3,.1,1.]
    data=model.createData();pin.framesForwardKinematics(model,data,q);obj=data.oMf[meta['object']['frame']]
    surfaces=[(0,'yaw1'),(0,'yaw2')];frames=[meta['frames'][grasp_frame(*s)]for s in surfaces]
    centers=np.array([data.oMf[f].translation.copy() for f in frames])
    points=[obj.actInv(p)for p in centers];normals=[obj.rotation.T@data.oMf[f].rotation[:,0]for f in frames]
    refs=GraspReferences(model,meta,q,q,surfaces,points,normals,
        samples=[(f,np.zeros((1,3)))for f in frames],posture_tracking=True)
    axis=centers[1]-centers[0];axis/=np.linalg.norm(axis);rotation=pin.exp3(.001*axis)
    moved=q.copy();moved[:3]=centers[0]+rotation@(q[:3]-centers[0]);moved[3:7]=pin.Quaternion(rotation).coeffs()
    refs.reference=moved.copy()
    pin.framesForwardKinematics(model,data,moved)
    for f,p in zip(frames,centers):np.testing.assert_allclose(data.oMf[f].translation,p,atol=1e-14)
    _,velocity=refs.update(moved,np.zeros(model.nv),q[oq:oq+7])
    assert velocity[3:6]@axis < -1e-5
    # Restoration uses the remaining freedom, preserving both grasp centers.
    body_velocity=velocity.copy();body_velocity[:3]=rotation.T@velocity[:3];body_velocity[3:6]=rotation.T@velocity[3:6]
    pin.forwardKinematics(model,data,moved,body_velocity);pin.updateFramePlacements(model,data)
    for f in frames:
        assert np.linalg.norm(pin.getFrameVelocity(model,data,f,pin.LOCAL_WORLD_ALIGNED).linear)<1e-7
