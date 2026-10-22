"""Optional CPU batch nearest-face query; distances use source float64 triangles."""
import numpy as np
import trimesh
import warp as wp
from warp._src.build import init_kernel_cache
from .model import ROOT


@wp.kernel
def _nearest_faces(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),faces:wp.array(dtype=int)):
    i=wp.tid();query=wp.mesh_query_point_no_sign(mesh,points[i],100.)
    faces[i]=query.face if query.result else -1


class WarpNearest:
    def __init__(self,mesh):
        cache=str(ROOT/'artifacts/warp_cache');wp.config.kernel_cache_dir=cache;wp.init()
        init_kernel_cache(cache)
        self.triangles=mesh.triangles
        self.mesh=wp.Mesh(points=wp.array(mesh.vertices,dtype=wp.vec3,device='cpu'),indices=wp.array(mesh.faces.ravel(),dtype=wp.int32,device='cpu'))
        # Compile/load once during known-geometry setup, outside the controller.
        self(np.asarray(mesh.vertices[:1]))

    def __call__(self,points):
        query=wp.array(points,dtype=wp.vec3,device='cpu');faces=wp.empty(len(points),dtype=int,device='cpu')
        wp.launch(_nearest_faces,len(points),[self.mesh.id,query,faces],device='cpu')
        indices=faces.numpy()
        if np.any(indices<0):raise ValueError('known mesh nearest face not found')
        nearest=trimesh.triangles.closest_point(self.triangles[indices],points)
        return nearest,np.linalg.norm(nearest-points,axis=1),indices
