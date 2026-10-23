"""Compare native transforms/bounds with the independent original Python path."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
import amsrr_det.grasp_geometry as geometry
import amsrr_det.native_grasp_geometry as native


@pytest.mark.parametrize('family',[None,'box','elongated_box','cylinder','ellipsoid','triangular_prism','tapered_box'])
@pytest.mark.parametrize('rotate',[False,True])
@pytest.mark.parametrize('opposed',[False,True])
def test_residual_and_derivative_match_python(monkeypatch,family,rotate,opposed):
    if opposed and family is None:return
    library=native.library()
    if library is None:pytest.skip('optional geometry kernel not built')
    shape=ObjectShape(family or 'box',[.38,.38 if family=='cylinder' else .25,.31],.7)
    model,meta=build_model(2,[dict(a=0,port_a='pitch1',b=1,port_b='yaw2')],shape.payload())
    q=pin.neutral(model);q[:3]=[.2,-.15,.5]
    q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(.2,-.3,.4)).coeffs()
    oq=meta['object']['q'];q[oq:oq+3]=[.1,.2,.5]
    q[oq+3:oq+7]=pin.Quaternion(pin.rpy.rpyToMatrix(-.1,.2,-.5)).coeffs()
    for i,a in enumerate(meta['actuators']):q[a['q']]=.04+.01*i
    surfaces=[(0,'yaw1'),(1,'pitch1')];data=model.createData();pin.framesForwardKinematics(model,data,q)
    frames=[meta['frames'][geometry.grasp_frame(*s)] for s in surfaces]
    points=np.array([data.oMf[f].translation for f in frames]);normals=np.array([data.oMf[f].rotation[:,0] for f in frames])
    evaluations=[]
    def inspect(fun,x0,*,jac,**kwargs):
        x=x0.copy();x[:3]+=[.03,-.04,.06]
        if rotate:x[3:6]=[.11,-.14,.08]
        evaluations.append((fun(x).copy(),jac(x).copy()))
        return SimpleNamespace(x=x,nfev=1)
    monkeypatch.setattr(geometry,'least_squares',inspect)
    for value in [None,library]:
        monkeypatch.setattr(native,'library',lambda:value)
        geometry.fit_contacts(model,meta,q,surfaces,points,normals,base_rotation=rotate,
            object_shape=shape if family else None,floor_clearance=1.,
            opposed_friction=4.5 if opposed else None,
            obstacles=[([.1,.2,.3],[.8,.7,.6]),([1.,-.3,.1],[.3,.2,.3])])
    np.testing.assert_allclose(evaluations[0][0],evaluations[1][0],atol=2e-14,rtol=2e-13)
    np.testing.assert_allclose(evaluations[0][1],evaluations[1][1],atol=5e-10,rtol=1e-9)


@pytest.mark.parametrize('size',[[.3,.4,.2],[.3,.3,.3],[.2,.2,.4]])
def test_ellipsoid_points_match_original_bracketed_solve(monkeypatch,size):
    library=native.library()
    if library is None:pytest.skip('optional geometry kernel not built')
    shape=ObjectShape('ellipsoid',size,.6);rng=np.random.default_rng(20261022)
    query=np.vstack([rng.uniform(-1,1,(500,3)),rng.uniform(-.08,.08,(500,3)),
                     np.zeros((1,3)),np.eye(3)*np.array(size)/2,np.eye(3)*.01])
    actual=shape.project_surface(query)
    monkeypatch.setattr(native,'library',lambda:None)
    expected=shape.project_surface(query)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=2e-10,rtol=2e-9)


@pytest.mark.parametrize('family',['triangular_prism','tapered_box'])
def test_polytope_points_match_original_triangle_queries(monkeypatch,family):
    if native.library() is None:pytest.skip('optional geometry kernel not built')
    shape=ObjectShape(family,[.3,.4,.2],.6);rng=np.random.default_rng(20261022)
    query=np.vstack([rng.uniform(-.6,.6,(1000,3)),shape.vertices,
                     shape.vertices+1e-7,shape.triangles.mean(axis=1)])
    actual=shape.project_surface(query)
    monkeypatch.setattr(native,'library',lambda:None)
    expected=shape.project_surface(query)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(actual[2],expected[2],atol=2e-8,rtol=2e-8)


def test_paired_triangle_projection_matches_original_material_query():
    if native.library() is None:pytest.skip('optional geometry kernel not built')
    import trimesh
    from amsrr_det.contact_geometry import material_triangles
    from amsrr_det.mujoco_model import CONTACT_CACHE
    rng=np.random.default_rng(20261022)
    for tri in [rng.normal(size=(1000,3,3)),
                *[material_triangles(p,str(CONTACT_CACHE)) for p in ['pitch1','pitch2','yaw1','yaw2']]]:
        points=tri.mean(axis=1)+rng.normal(size=(len(tri),3))*.01
        expected=trimesh.triangles.closest_point(tri,points)
        np.testing.assert_allclose(native.project_to_triangles(tri,points),expected,atol=3e-14,rtol=1e-12)
