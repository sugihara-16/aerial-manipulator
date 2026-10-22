"""Build reusable unit-shape geometry; contains no assembled robot candidates."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from amsrr_det.convex_parts import split_convex
from amsrr_det.model import ROOT


def build(subdivisions=5):
    started=time.perf_counter();mesh=trimesh.creation.icosphere(subdivisions=subdivisions)
    hull=ConvexHull(mesh.vertices);pieces=split_convex(mesh.vertices)
    assert all(len(v)<=120 and len(f)<=255 for v,f in pieces)
    volume=sum(ConvexHull(v).volume for v,_ in pieces)
    if not np.isclose(volume,hull.volume,rtol=1e-10):raise ValueError('partition changed unit-shape volume')
    path=ROOT/'build/ellipsoid_unit_parts.npz'
    np.savez_compressed(path,**{key+str(i):array for i,piece in enumerate(pieces) for key,array in zip(['v','f'],piece)})
    # r*B is inside the unit polyhedron P, which is inside B. After scaling A,
    # Hausdorff(A*P,A*B) <= norm(A,2)*(1-r). Cooking/float error is separate.
    radial_error=float(1-np.min(-hull.equations[:,3]/np.linalg.norm(hull.equations[:,:3],axis=1)))
    manifest=dict(subdivisions=subdivisions,parts=len(pieces),radial_error_bound_unit=radial_error,
                  kind='common unit shape geometry, not robot structure candidates',
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256((ROOT/'amsrr_det/convex_parts.py').read_bytes()).hexdigest(),trimesh_version=trimesh.__version__,wall_s=time.perf_counter()-started)
    path.with_suffix('.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':print(json.dumps(build(),indent=2))
