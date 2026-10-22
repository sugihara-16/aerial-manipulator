import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import ROOT, build_model, validate_graph, DockEdge


def test_single_module_matches_independent_urdf_parser():
    model, meta = build_model()
    ref = pin.buildModelFromUrdf(str(ROOT/'assets/robots/holon/holon.urdf'),pin.JointModelFreeFlyer(),"free_base")
    locks=[ref.getJointId('rotor'+str(i)) for i in range(1,5)]
    ref=pin.buildReducedModel(ref,locks,pin.neutral(ref))
    ref.gravity=model.gravity
    rng=np.random.default_rng(104)
    for _ in range(12):
        q=pin.neutral(model); qr=pin.neutral(ref)
        qr[:7]=q[:7]=[.3,-.2,1.,.1,.2,-.1,np.sqrt(.94)]
        velocity=rng.normal(0,.1,model.nv);vr=np.zeros(ref.nv);vr[:6]=velocity[:6]
        for a in meta['actuators']:
            j=ref.joints[ref.getJointId(a['name'].split('/')[1])]
            qr[j.idx_q]=q[a['q']]=rng.uniform(-.4,.4)
            vr[j.idx_v]=velocity[a['v']]
            ref.armature[j.idx_v]=model.armature[a['v']]
        d=model.createData();dr=ref.createData()
        pin.forwardKinematics(model,d,q,velocity);pin.updateFramePlacements(model,d)
        pin.forwardKinematics(ref,dr,qr,vr);pin.updateFramePlacements(ref,dr)
        for name,fid in meta['frames'].items():
            np.testing.assert_allclose(d.oMf[fid].homogeneous,dr.oMf[ref.getFrameId(name.split('/')[1])].homogeneous,atol=2e-14)
        np.testing.assert_allclose(pin.centerOfMass(model,d,q),pin.centerOfMass(ref,dr,qr),atol=2e-14)
        # Inverse dynamics compares both inertia and velocity-dependent terms.
        tau=pin.rnea(model,d,q,velocity,np.zeros(model.nv));tr=pin.rnea(ref,dr,qr,vr,np.zeros(ref.nv))
        np.testing.assert_allclose(tau[:6],tr[:6],atol=2e-13)
        for a in meta['actuators']:
            assert tau[a['v']]==pytest.approx(tr[ref.joints[ref.getJointId(a['name'].split('/')[1])].idx_v],abs=2e-13)


@pytest.mark.parametrize('n',[1,3,8])
def test_mass_and_degrees_of_freedom_survive_assembly(n):
    edges=[DockEdge(i-1,'pitch1',i,'yaw2') for i in range(1,n)]
    model,meta=build_model(n,edges)
    assert model.nv==6+8*n
    assert meta['mass']==pytest.approx(n*1.8667403010993027,rel=1e-13)
    q=pin.neutral(model)
    for a in meta['actuators']:q[a['q']]=.17
    d=model.createData();pin.framesForwardKinematics(model,d,q)
    mate=pin.SE3(pin.rpy.rpyToMatrix(0.,0.,np.pi),np.zeros(3))
    for e in edges:
        A=d.oMf[meta['frames'][f'm{e.a}/pitch_connect_dummy_1']]
        B=d.oMf[meta['frames'][f'm{e.b}/yaw_connect_dummy_2']]
        np.testing.assert_allclose((A.inverse()*B).homogeneous,mate.homogeneous,atol=5e-14)


def test_incoming_joint_reversal_preserves_child_module_geometry():
    model,meta=build_model(2,[DockEdge(0,'pitch1',1,'yaw2')])
    single,sm=build_model();q=pin.neutral(model);qs=pin.neutral(single)
    for a in meta['actuators']:
        q[a['q']]=.23 if a['role']=='dock' else -.12
    for a in sm['actuators']:qs[a['q']]=.23 if a['role']=='dock' else -.12
    d=model.createData();ds=single.createData();pin.framesForwardKinematics(model,d,q);pin.framesForwardKinematics(single,ds,qs)
    root=d.oMf[meta['frames']['m1/root']]
    for name,fid in sm['frames'].items():
        np.testing.assert_allclose((root.inverse()*d.oMf[meta['frames']['m1/'+name.split('/')[1]]]).homogeneous,ds.oMf[fid].homogeneous,atol=4e-14)


def test_declared_gravity_and_payload_inertia_are_preserved():
    gravity=[.1,.2,-9.79]
    payload=dict(mass=1.,size=[.3,.4,.2],com=[.01,-.02,.005],inertia=[[.03,.001,0],[.001,.04,.002],[0,.002,.05]])
    model,meta=build_model(payload=payload,gravity=gravity)
    np.testing.assert_array_equal(model.gravity.linear,gravity)
    physical=model.inertias[meta['object']['joint']]
    np.testing.assert_allclose(physical.lever,payload['com'])
    np.testing.assert_allclose(physical.inertia,payload['inertia'])


@pytest.mark.parametrize('mass,inertia',[(float('nan'),np.eye(3)),(1.,np.diag([1.,1.,3.])),(1.,np.diag([-1.,1.,1.]))])
def test_nonphysical_payload_is_rejected(mass,inertia):
    with pytest.raises(ValueError):build_model(payload=dict(mass=mass,size=[.3,.4,.2],inertia=inertia))


@pytest.mark.parametrize('edges',[
    [DockEdge(0,'pitch1',0,'yaw2')],
    [DockEdge(0,'pitch1',1,'yaw2'),DockEdge(0,'pitch1',2,'yaw2')],
    [DockEdge(0,'pitch1',1,'pitch2')],
    [DockEdge(0,'yaw1',1,'yaw2')],
    [DockEdge(0,'pitch1',1,'yaw2'),DockEdge(1,'pitch1',2,'yaw2'),DockEdge(2,'pitch1',0,'yaw2')]])
def test_invalid_connections_rejected(edges):
    with pytest.raises(ValueError):validate_graph(3,edges)
