import numpy as np
import pytest
import trimesh
pytest.importorskip('warp')
from amsrr_det.mesh_surface import MeshSurface
from amsrr_det.objects import ObjectShape


@pytest.mark.parametrize('mesh',[trimesh.creation.icosphere(subdivisions=2,radius=.15),trimesh.creation.annulus(r_min=.06,r_max=.15,height=.1,sections=48)])
def test_cpu_batch_signed_distance_agrees_at_interior_exterior_and_near_surface(mesh):
    nominal=ObjectShape('box',[.3,.3,.3],.6)
    old=MeshSurface(nominal,mesh);new=MeshSurface(nominal,mesh,query_backend='warp')
    centers=mesh.triangles_center[::5];normals=mesh.face_normals[::5]
    points=np.vstack([np.random.default_rng(16).uniform(-.2,.2,(500,3)),*(centers+d*normals for d in [-1e-6,-1e-7,1e-7,1e-6])])
    a,b=old.project_surface(points),new.project_surface(points)
    np.testing.assert_array_equal(a[1]<0,b[1]<0)
    np.testing.assert_allclose(a[1],b[1],atol=1e-8,rtol=1e-7)
    assert np.isfinite(b[0]).all() and np.isfinite(b[2]).all()
