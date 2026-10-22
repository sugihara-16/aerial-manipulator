import numpy as np
import pytest
import trimesh
from amsrr_det.objects import ObjectShape


@pytest.mark.parametrize('family',['triangular_prism','tapered_box'])
def test_fast_face_projection_matches_all_triangle_reference(family):
    shape=ObjectShape(family,[.30,.42,.24],.6)
    x=np.random.default_rng(194).uniform(-.4,.4,(500,3))
    triangles=shape.vertices[shape.faces]
    full=trimesh.triangles.closest_point(np.broadcast_to(triangles,(len(x),*triangles.shape)).reshape(-1,3,3),np.repeat(x,len(triangles),axis=0)).reshape(len(x),-1,3)
    distances=np.linalg.norm(full-x[:,None,:],axis=2)
    expected=full[np.arange(len(x)),np.argmin(distances,axis=1)]
    points,gaps,normals=shape.project_surface(x)
    np.testing.assert_allclose(points,expected,atol=1e-12)
    np.testing.assert_allclose(abs(gaps),np.min(distances,axis=1),atol=1e-12)
