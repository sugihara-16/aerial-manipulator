"""Geometric reference protection, independently checked by forward kinematics."""
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.grasp_control import GraspReferences,surface_points
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.objects import ObjectShape


def fixture():
    model,meta=build_model(payload=dict(size=[.3,.3,.3],mass=.6));q=pin.neutral(model)
    q[2]=q[meta['object']['q']+2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q);obj=data.oMf[meta['object']['frame']]
    surfaces=[(0,'yaw1'),(0,'yaw2'),(0,'pitch1')];frames=[meta['frames'][grasp_frame(*s)] for s in surfaces]
    points=[obj.actInv(data.oMf[f].translation) for f in frames]
    normals=np.array([obj.rotation.T@data.oMf[f].rotation[:,0] for f in frames])
    def make():return GraspReferences(model,meta,q,q,surfaces,points,normals,samples=[(f,np.zeros((1,3))) for f in frames])
    return model,meta,q,frames,np.array(points),normals,make


def test_preload_request_survives_opposing_object_motion_without_relaxing_bounds():
    m,meta,q,frames,points,normals,make=fixture();goal=q[meta['object']['q']:].copy();goal[:3]-=.004*normals[2]
    old=make();guard=make();preload=np.array([0.,0.,.00015]);v=np.zeros(m.nv)
    old.update(q,v,goal,preload=preload)
    before=guard.reference.copy();_,velocity=guard.update(q,v,goal,preload=preload,preserve_preload=True)
    assert guard.last_diagnostics['preload_reference_feasible']
    gamma=guard.last_diagnostics['required_closing_rate_m_s'][0];assert gamma>0
    data=m.createData()
    def rate(nu):
        step=np.zeros(m.nv);step[:6]=nu[:6];step[guard.vi]=nu[6:]
        pin.framesForwardKinematics(m,data,pin.integrate(m,q,.001*step))
        return float(normals[2]@(data.oMf[frames[2]].translation-(points[2]+q[meta['object']['q']:meta['object']['q']+3]))/.001)
    assert rate(old.last_velocity)<0
    assert rate(guard.last_velocity)>=gamma-1e-8
    assert np.all(abs(guard.last_velocity)<=np.r_[[.001]*3,[.003]*3,[.005]*len(guard.vi)]+1e-12)
    assert np.linalg.norm(guard.reference-before)>0


def test_unreachable_object_speed_is_rejected_without_advancing_reference():
    m,meta,q,frames,points,normals,make=fixture();refs=make();v=np.zeros(m.nv)
    v[meta['object']['v']:meta['object']['v']+3]=normals[2]
    before=refs.reference.copy()
    target,_=refs.update(q,v,q[meta['object']['q']:],preload=[0.,0.,.00015],preserve_preload=True)
    assert not refs.last_diagnostics['preload_reference_feasible']
    np.testing.assert_array_equal(target,before);np.testing.assert_array_equal(refs.reference,before)


def test_no_deficit_keeps_the_original_reference_exactly():
    m,meta,q,frames,points,normals,make=fixture();first,second=make(),make();goal=q[meta['object']['q']:].copy();goal[0]+=.001
    a=first.update(q,np.zeros(m.nv),goal,preload=[-.001]*3)
    b=second.update(q,np.zeros(m.nv),goal,preload=[-.001]*3,preserve_preload=True)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_real_surface_compression_gradient_matches_refreshed_geometry():
    shape=ObjectShape('box',[.2,.2,.2],.6);m,meta=build_model(payload=shape.payload());q=pin.neutral(m);q[2]=1.
    data=m.createData();pin.framesForwardKinematics(m,data,q);surface=(0,'pitch1');frame,points=surface_points(m,meta,q,surface)
    transform=data.oMf[frame];oq=meta['object']['q']
    q[oq:oq+3]=transform.act(np.array([points[:,0].max()+.1-.0001,0,0]))
    q[oq+3:oq+7]=pin.Quaternion(transform.rotation@pin.exp3(np.array([0.,.002,.001]))).coeffs()
    def make(pose):return GraspReferences(m,meta,pose,pose,[surface],[[0,0,0]],[[1,0,0]],shape=shape)
    refs=make(q);J,H,N,depth=refs.geometry(q);nu=np.zeros(refs.size);nu[:3]=[.0003,-.0002,.0001];nu[6:]=np.linspace(-.001,.001,len(refs.vi))
    velocity=np.zeros(m.nv);velocity[:6]=nu[:6];velocity[refs.vi]=nu[6:]
    dt=1e-4;moved=pin.integrate(m,q,dt*velocity);new_depth=make(moved).geometry(moved)[3]
    np.testing.assert_allclose((new_depth-depth)/dt,refs.compression_map@J@nu,atol=2e-8,rtol=1e-4)
