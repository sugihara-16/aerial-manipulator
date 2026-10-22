import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from amsrr_det.convex_parts import split_convex


def test_convex_partition_keeps_volume_and_support():
    mesh=trimesh.creation.icosphere(subdivisions=3);vertices=mesh.vertices*np.array([.2,.13,.07])
    pieces=split_convex(vertices);original=ConvexHull(vertices)
    assert len(pieces)>1 and all(len(v)<=120 and len(f)<=255 for v,f in pieces)
    assert abs(sum(ConvexHull(v).volume for v,_ in pieces)-original.volume)<1e-10
    directions=np.random.default_rng(13).normal(size=(200,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    np.testing.assert_allclose(np.max(vertices@directions.T,axis=0),np.max(np.concatenate([v for v,_ in pieces])@directions.T,axis=0),atol=1e-10)
    for v,f in pieces:
        solid=trimesh.Trimesh(v,f,process=False)
        assert solid.is_watertight and solid.volume>0
