"""Check analytic contact/environment derivatives against independent differences."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import pytest
import amsrr_det.grasp_geometry as geometry
from amsrr_det.model import build_model


@pytest.mark.parametrize('family',['box','ellipsoid','cylinder','triangular_prism','tapered_box'])
@pytest.mark.parametrize('normal_moments',[False,True])
@pytest.mark.parametrize('surface_count',[1,2])
def test_surface_sliding_derivative_against_whole_configuration_difference(monkeypatch,family,normal_moments,surface_count):
    from amsrr_det.objects import ObjectShape
    shape=ObjectShape(family,[.3,.3,.2],.6)
    m,meta=build_model(payload=shape.payload());q=pin.neutral(m);q[:3]=[.2,-.15,.5]
    oq=meta['object']['q'];q[oq:oq+3]=q[:3]+[.08,.07,.04]
    q[oq+3:oq+7]=pin.Quaternion(pin.rpy.rpyToMatrix(.1,-.2,.3)).coeffs()
    surfaces=[(0,'yaw1'),(0,'yaw2')][:surface_count];data=m.createData();pin.framesForwardKinematics(m,data,q)
    frames=[meta['frames'][geometry.grasp_frame(*s)] for s in surfaces]
    points=np.array([data.oMf[f].translation for f in frames]);normals=np.array([data.oMf[f].rotation[:,0] for f in frames])
    def inspect(fun,x0,*,jac,**kwargs):
        x=x0.copy();x[3:6]=[.13,-.08,.05];epsilon=1e-7
        numerical=np.column_stack([(fun(x+np.eye(len(x))[i]*epsilon)-fun(x-np.eye(len(x))[i]*epsilon))/(2*epsilon) for i in range(len(x))])
        np.testing.assert_allclose(jac(x),numerical,atol=2e-6,rtol=2e-5)
        return SimpleNamespace(x=x0,nfev=0)
    monkeypatch.setattr(geometry,'least_squares',inspect)
    extra=dict(normal_moment_axis=[0.,0.,1.],normal_moment_samples=[np.array([[.002,.004,-.001],[.001,-.002,.003]]) for _ in surfaces]) if normal_moments else {}
    geometry.fit_contacts(m,meta,q,surfaces,points,normals,base_rotation=True,object_shape=shape,**extra)


def test_pose_normal_and_environment_derivatives(monkeypatch):
    m,meta=build_model(2,[dict(a=0,port_a='pitch1',b=1,port_b='yaw2')])
    q=pin.neutral(m);q[:3]=[.2,-.15,.5]
    for i,a in enumerate(meta['actuators']):q[a['q']]=.05+.01*i
    surfaces=[(0,'yaw1'),(1,'pitch1')];data=m.createData();pin.framesForwardKinematics(m,data,q)
    frames=[meta['frames'][geometry.grasp_frame(*s)] for s in surfaces]
    points=np.array([data.oMf[f].translation for f in frames]);normals=np.array([data.oMf[f].rotation[:,0] for f in frames])
    def inspect(fun,x0,*,jac,**kwargs):
        x=x0.copy();x[3:6]=[.13,-.08,.05]
        analytical=jac(x);epsilon=1e-7
        numerical=np.column_stack([(fun(x+np.eye(len(x))[i]*epsilon)-fun(x-np.eye(len(x))[i]*epsilon))/(2*epsilon) for i in range(len(x))])
        np.testing.assert_allclose(analytical,numerical,atol=2e-7,rtol=2e-5)
        return SimpleNamespace(x=x0,nfev=0)
    monkeypatch.setattr(geometry,'least_squares',inspect)
    geometry.fit_contacts(m,meta,q,surfaces,points,normals,base_rotation=True,floor_clearance=2.,
                          obstacles=[(np.array([.15,.2,.3]),np.array([.8,.7,.6]))])


@pytest.mark.parametrize('fixed_world',[False,True])
def test_object_collision_repair_derivative_in_active_constraint(monkeypatch,fixed_world):
    import coal
    from amsrr_det.objects import ObjectShape
    shape=ObjectShape('box',[.15,.17,.19],.6)
    m,meta=build_model(payload=shape.payload());q=pin.neutral(m);q[:3]=[.2,-.15,.5]
    oq=meta['object']['q'];q[oq:oq+3]=q[:3]+[.12,.015,.01]
    surfaces=[(0,'yaw1')];data=m.createData();pin.framesForwardKinematics(m,data,q)
    frame=meta['frames'][geometry.grasp_frame(*surfaces[0])]
    points=np.array([data.oMf[frame].translation]);normals=np.array([data.oMf[frame].rotation[:,0]])
    def inspect(fun,x0,*,jac,**kwargs):
        x=x0.copy();x[3:6]=[.13,-.08,.05]
        analytical=jac(x);epsilon=1e-6
        numerical=np.column_stack([(fun(x+np.eye(len(x))[i]*epsilon)-fun(x-np.eye(len(x))[i]*epsilon))/(2*epsilon) for i in range(len(x))])
        # Six site residuals precede the active collision-distance residual.
        assert fun(x)[6]>.01 and np.linalg.norm(analytical[6])>1.
        np.testing.assert_allclose(analytical,numerical,atol=2e-5,rtol=2e-4)
        return SimpleNamespace(x=x0,nfev=0)
    monkeypatch.setattr(geometry,'least_squares',inspect)
    name='fixed_support' if fixed_world else 'object'
    transform=data.oMf[meta['object']['frame']] if fixed_world else pin.SE3.Identity()
    geometry.fit_contacts(m,meta,q,surfaces,points,normals,base_rotation=True,
        avoid_pairs=[('m0/main_body',name)],avoid_shapes={name:(transform,coal.Box(*shape.size))})
