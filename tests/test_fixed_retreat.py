import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.grasp_control import GraspReferences
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.retreat import FixedRetreat,rest_to_rest_durations


def test_finite_retreat_has_fixed_endpoint_and_ignores_object_velocity():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    frame=meta['frames'][grasp_frame(0,'yaw1')];obj=data.oMf[meta['object']['frame']]
    points=[obj.actInv(data.oMf[frame].translation)];normals=[obj.rotation.T@data.oMf[frame].rotation[:,0]]
    def make():
        grasp=GraspReferences(model,meta,q,q,[(0,'yaw1')],points,normals,samples=[(frame,np.zeros((1,3)))])
        return grasp,FixedRetreat(grasp,q)
    first,a=make();second,b=make();v=np.zeros(model.nv);moving=v.copy();moving[meta['object']['v']:meta['object']['v']+6]=[.01,0,0,0,.02,0]
    before=a.at(0.)
    np.testing.assert_array_equal(before,q)
    ra,va=a.update(q,v,q[meta['object']['q']:],preload=[-.002])
    rb,vb=b.update(q,moving,q[meta['object']['q']:],preload=[-.002])
    np.testing.assert_array_equal(ra,rb);np.testing.assert_array_equal(va,vb)
    assert a.distance<=.001*first.dt+1e-12
    endpoint=a.at(.03);pin.framesForwardKinematics(model,data,endpoint)
    end=data.oMf[frame].translation.copy();pin.framesForwardKinematics(model,data,q)
    np.testing.assert_allclose(end,data.oMf[frame].translation-.03*data.oMf[frame].rotation[:,0],atol=1e-5)
    np.testing.assert_array_equal(a.at(1.),endpoint)
    first.last_velocity[6]=.01
    bridge=FixedRetreat(first,q);previous=first.last_velocity.copy()
    limits=np.r_[np.full(3,.1),np.full(3,.3),np.full(len(first.docks),.5)]
    for step in range(300):
        # The requested progression stops part way through. Braking must not
        # turn this into an instantaneous change in the drive velocity target.
        preload=[-.002] if step<80 else [0.]
        bridge.update(q,v,q[meta['object']['q']:],preload=preload)
        assert np.all(abs(first.last_velocity-previous)<=limits*first.dt+1e-5)
        previous=first.last_velocity.copy()
    np.testing.assert_allclose(first.last_velocity,0.,atol=1e-10)


def test_loaded_reference_offset_is_included_in_material_clearance_goal():
    from amsrr_det.objects import ObjectShape
    from amsrr_det.grasp_control import surface_points
    shape=ObjectShape('box',[.3,.3,.3],.6)
    model,meta=build_model(1,payload=shape.payload());q=pin.neutral(model);q[2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    frame,points=surface_points(model,meta,q,(0,'yaw1'));pad=data.oMf[frame].copy();oq=meta['object']['q']
    q[oq:oq+3]=pad.act(np.array([points[:,0].max()+.15,0.,0.]))
    q[oq+3:oq+7]=pin.Quaternion(pad.rotation).coeffs()
    reference=q.copy();reference[:3]+=.004*pad.rotation[:,0]
    grasp=GraspReferences(model,meta,q,reference,[(0,'yaw1')],[[-.15,0,0]],[[1,0,0]],shape=shape)
    path=FixedRetreat(grasp,q)
    assert path.initial_reference_gaps[0]<-.0039
    assert path.distances[1]>.0059
    final=path.path[1]
    assert -grasp.geometry(final)[-1][0]>=.00199
    # A small frozen endpoint can intersect an object that settled elsewhere.
    moved=final.copy();moved[oq:oq+3]-=.01*pad.rotation[:,0]
    assert -grasp.geometry(moved)[-1][0]<0.
    # The opening phase can reach the final finite goal directly; it need not
    # already have separated before it is allowed to request that endpoint.
    wide=FixedRetreat(grasp,q,opening_gap_m=.02,final_gap_m=.02)
    assert wide.opening_distance==wide.distances[-1]
    moved=wide.path[-1].copy();moved[oq:oq+3]-=.01*pad.rotation[:,0]
    assert -grasp.geometry(moved)[-1][0]>=.00999


def test_reversing_retreat_corner_has_bounded_velocity_change_and_same_geometry():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    frame=meta['frames'][grasp_frame(0,'yaw1')];obj=data.oMf[meta['object']['frame']]
    points=[obj.actInv(data.oMf[frame].translation)];normals=[obj.rotation.T@data.oMf[frame].rotation[:,0]]
    grasp=GraspReferences(model,meta,q,q,[(0,'yaw1')],points,normals,samples=[(frame,np.zeros((1,3)))])
    first=q.copy();second=q.copy();third=q.copy()
    second[:3]+=[.02,.01,.015];third[:3]+=[-.01,.03,.02]
    second[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.03,-.02,.05)).coeffs()
    third[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(-.02,.04,-.03)).coeffs()
    for i,a in enumerate(meta['actuators']):
        second[a['q']]=.12*np.cos(i);third[a['q']]=-.08*np.cos(i)
    path=[first,second,third];distances=np.array([0.,.01,.02])
    durations=rest_to_rest_durations(model,meta,path,distances)
    retreat=FixedRetreat(grasp,q,prepared=dict(path=path,distances=distances,segment_durations=durations))
    previous=np.zeros(model.nv)
    for _ in range(int(np.ceil(sum(durations)/grasp.dt))+3):
        target,velocity=retreat.update(q,np.zeros(model.nv),q[meta['object']['q']:],preload=[-.03],normal_speed=.006)
        change=(velocity-previous)/grasp.dt
        assert np.linalg.norm(velocity[:3])<=.06+1e-8
        assert np.linalg.norm(velocity[3:6])<=.15+1e-8
        assert np.linalg.norm(change[:3])<=.1+1e-8
        assert np.linalg.norm(change[3:6])<=.3+1e-8
        for a in meta['actuators']:
            assert abs(velocity[a['v']])<=(.2 if a['role']=='dock' else .3)+1e-8
            assert abs(change[a['v']])<=.5+1e-8
        # Retiming must keep every point on the already checked pose path.
        np.testing.assert_allclose(target,retreat.at(retreat.distance),atol=1e-12)
        previous=velocity
    np.testing.assert_allclose(target,third,atol=1e-12)
    np.testing.assert_allclose(velocity,0.,atol=1e-12)


def test_rigid_withdrawal_uses_free_flight_speed_and_keeps_base_limits():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    first=pin.neutral(model);first[2]=1.
    opened=first.copy();opened[0]+=.02
    raised=opened.copy();raised[2]+=.6
    path=[first,opened,raised];distances=np.array([0.,.02,.62])
    durations=rest_to_rest_durations(model,meta,path,distances,normal_speed=[.006,.06])
    from amsrr_det.numerics import quintic_reference
    peaks=[]
    for a,b,duration in zip(path[:-1],path[1:],durations):
        speed=[]
        for t in np.linspace(0.,duration,101):
            _,v,acceleration,_=quintic_reference(model,meta,a,b,t,duration)
            speed.append(np.linalg.norm(v[:3]))
            assert np.linalg.norm(v[:3])<=.06+1e-12
            assert np.linalg.norm(acceleration[:3])<=.1+1e-12
        peaks.append(max(speed))
    np.testing.assert_allclose(peaks,[.006,.06],atol=1e-12)
    assert durations[1]<20.
