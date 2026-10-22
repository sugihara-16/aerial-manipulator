import numpy as np
import pinocchio as pin
import pytest
import trimesh
from amsrr_det.objects import ObjectShape
from amsrr_det.mujoco_model import make_plant


@pytest.mark.parametrize('family',['box','elongated_box','cylinder','ellipsoid','triangular_prism','tapered_box'])
def test_object_mass_inertia_and_resting_height(family):
    shape=ObjectShape(family,[.3,.3,.2],.6)
    plant=make_plant(1,payload=shape.payload(),meshes=False)
    obj=plant.meta['object'];inertia=plant.pin.inertias[obj['joint']]
    assert inertia.mass==pytest.approx(.6)
    np.testing.assert_allclose(inertia.inertia,shape.inertia,atol=1e-13)
    assert np.linalg.eigvalsh(shape.inertia).min()>0
    quat=pin.Quaternion(pin.rpy.rpyToMatrix(np.pi/2,0,.7)).coeffs()
    pose=np.r_[.2,-.3,0.,quat];pose[2]=.2-shape.lowest_z(pose)
    assert shape.lowest_z(pose)==pytest.approx(.2)
    if shape.vertices is not None:
        mesh=trimesh.Trimesh(shape.vertices,shape.faces,process=False)
        np.testing.assert_allclose(mesh.center_mass,0,atol=1e-14)
        np.testing.assert_allclose(mesh.moment_inertia*.6/mesh.mass,shape.inertia,atol=1e-13)
    for direction in [[1,0,0],[0,1,0],[.2,.3,.7]]:
        point,normal=shape.ray_surface(direction)
        assert normal@direction>0
        np.testing.assert_allclose(np.cross(point,direction),0,atol=1e-13)
        assert np.linalg.norm(normal)==pytest.approx(1.)


@pytest.mark.parametrize('family,size',[
    ('box',[.3,.4,.2]),('ellipsoid',[.3,.4,.2]),('cylinder',[.3,.3,.2]),
    ('triangular_prism',[.3,.4,.2]),('tapered_box',[.3,.4,.2])])
def test_nearest_surface_is_real_and_has_consistent_signed_gap(family,size):
    shape=ObjectShape(family,size,.6)
    rng=np.random.default_rng(194);directions=rng.normal(size=(40,3))
    for direction in directions:
        point,normal=shape.ray_surface(direction)
        # Stay off nonsmooth vertices: the chosen supporting normal produces an
        # exact nearest point for the exterior displacement.
        queries=np.array([point+.0004*normal,point+.04*normal])
        surface,gap,n=shape.project_surface(queries)
        np.testing.assert_allclose(surface,np.tile(point,(2,1)),atol=1e-8)
        np.testing.assert_allclose(gap,[.0004,.04],atol=1e-8)
        np.testing.assert_allclose(n,np.tile(normal,(2,1)),atol=1e-8)
    p,g,n=shape.project_surface([[0,0,0]])
    assert g[0]<0 and np.isfinite(p).all()
    np.testing.assert_allclose(np.linalg.norm(n,axis=1),1.,atol=1e-8)
