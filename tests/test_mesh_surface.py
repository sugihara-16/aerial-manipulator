import numpy as np
import trimesh
from amsrr_det.mesh_surface import MeshSurface
from amsrr_det.objects import ObjectShape


def test_known_closed_surface_matches_independent_box_queries():
    nominal=ObjectShape('box',np.array([.3,.4,.2]),.6)
    mesh=trimesh.creation.box(extents=nominal.size)
    actual=MeshSurface(nominal,mesh)
    points=np.array([[.16,0,0],[.14,0,0],[.16,.21,.11],[.15,0,0],[0,0,0],[.14,.19,.09]])
    p,gap,n=actual.project_surface(points);expected,eg,en=nominal.project_surface(points)
    np.testing.assert_allclose(gap,eg,atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(points-p,axis=1),abs(gap),atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(n,axis=1),1,atol=1e-10)
    assert actual.lowest_z(np.array([0,0,.3,0,0,0,1]))==.19999999999999998


def test_union_containment_and_query_ignore_internal_partition_face():
    nominal=ObjectShape('box',np.array([.3,.4,.2]),.6)
    pieces=[]
    for x in [-.075,.075]:
        part=trimesh.creation.box(extents=[.15,.4,.2]);part.apply_translation([x,0,0]);pieces.append(part)
    actual=MeshSurface(nominal,trimesh.creation.box(extents=nominal.size),pieces)
    points=np.array([[0,0,.09],[0,0,.11],[.151,0,0],[0,0,0]])
    np.testing.assert_array_equal(actual.contains(points),[True,False,False,True])
    _,gaps,_=actual.project_surface(points)
    np.testing.assert_allclose(gaps,[-.01,.01,.001,-.1],atol=1e-10)
