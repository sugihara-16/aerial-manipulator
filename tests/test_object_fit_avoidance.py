import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
from amsrr_det.native_grasp_geometry import GeometryKernel,library

@pytest.mark.parametrize('analytic',[False,True])
def test_object_avoidance_fit_without_native_kernel(monkeypatch,analytic):
    import amsrr_det.grasp_geometry as geometry
    import amsrr_det.native_grasp_geometry as native
    monkeypatch.setattr(native,'library',lambda:None)
    shape=ObjectShape('box',[.5,.4,.3],1.)
    model,meta=build_model(1,payload=shape.payload());q=pin.neutral(model)
    q[meta['object']['q']:meta['object']['q']+3]=[.5,0.,0.]
    fitted,report=geometry.fit_contacts(model,meta,q,[(0,'yaw1')],
        np.array([[.25,0.,0.]]),np.array([[1.,0.,0.]]),
        object_shape=shape,avoid_object=True,analytic_jacobian=analytic,max_evaluations=2)
    assert np.all(np.isfinite(fitted))
    assert np.isfinite(report['position_error_m'])

@pytest.mark.parametrize('family',['box','cylinder','ellipsoid','triangular_prism','tapered_box'])
def test_authored_sample_penalty_and_derivative_match_independent_surface_queries(family):
    assert library() is not None
    shape=ObjectShape(family,[1.1,1.1 if family=='cylinder' else .8,.7],4.)
    model,meta=build_model(1,payload=shape.payload());q=pin.neutral(model)
    obj=meta['object'];q[obj['q']:obj['q']+3]=[.073,-.042,.031]
    q[obj['q']+3:obj['q']+7]=pin.Quaternion(pin.rpy.rpyToMatrix(.2,-.3,.1)).coeffs()
    frame=meta['frames']['m0/main_body'];of=meta['object']['frame']
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    boundary=[]
    for direction in np.r_[np.eye(3),-np.eye(3)]:
        point,normal=shape.ray_surface(direction)
        for gap in [.03-1e-9,.03,.03+1e-9]:
            boundary.append(data.oMf[frame].actInv(data.oMf[of].act(point+gap*normal)))
    points=np.vstack([[[.12,.14,.08],[-.13,.07,-.14],[.7,.6,.8]],
                      np.random.default_rng(87).uniform(-1.5,1.5,(128,3)),boundary])
    samples=(np.full(len(points),frame),points)
    lift=np.zeros((model.nv,3));lift[:3]=np.eye(3)
    kernel=GeometryKernel(model,[of],[],[],None,3)
    kernel.evaluate(q,lift);r,D=kernel.object_avoidance(lift,of,samples,shape)
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    local=np.array([data.oMf[of].actInv(data.oMf[frame].act(p))for p in points])
    grouped,grouped_D=kernel.object_avoidance(lift,of,samples,shape,grouped=True)
    worst=int(np.argmax(r))
    np.testing.assert_allclose(grouped,[r[worst]],atol=1e-12)
    np.testing.assert_allclose(grouped_D,D[[worst]],atol=1e-12)
    expected=2*np.maximum(0,.03-shape.project_surface(local)[1])
    np.testing.assert_allclose(r,expected,atol=1e-11)
    numeric=[]
    for j in range(3):
        vals=[]
        for sign in [1.,-1.]:
            trial=q.copy();trial[j]+=sign*1e-6;kernel.evaluate(trial,lift)
            vals.append(kernel.object_avoidance(lift,of,samples,shape)[0])
        numeric.append((vals[0]-vals[1])/2e-6)
    # The boundary probes deliberately straddle the max(0, 3 cm-gap) kink;
    # a central difference there is not its one-sided active derivative.
    # Their residuals were checked above. Compare derivatives off that kink.
    ordinary=len(points)-len(boundary)
    np.testing.assert_allclose(D[:ordinary],np.array(numeric).T[:ordinary],atol=3e-6)
