"""Optional exact frame/bound batching; known object geometry stays in Python."""
import ctypes as ct
from functools import lru_cache
import hashlib,json
import numpy as np
from .model import ROOT


@lru_cache(maxsize=1)
def library():
    folder=ROOT/'build/geometry_kernel'
    try:
        identity=json.loads((folder/'identity.json').read_text())
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=v for p,v in identity['sources'].items()):return None
        binary=folder/'libgeometry_kernel.so'
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=identity['binary_sha256']:return None
    except (OSError,KeyError,ValueError):return None
    lib=ct.CDLL(str(binary));lib.geometry_error.restype=ct.c_char_p
    lib.geometry_create.argtypes=[ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_double]
    lib.geometry_create.restype=ct.c_void_p
    lib.geometry_destroy.argtypes=[ct.c_void_p]
    lib.geometry_evaluate.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p]
    lib.geometry_evaluate.restype=ct.c_int
    lib.geometry_fit_evaluate.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,
        ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,
        ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_void_p]
    lib.geometry_fit_evaluate.restype=ct.c_int
    lib.ellipsoid_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
    lib.ellipsoid_points.restype=None
    lib.nearest_triangle_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_void_p]
    lib.nearest_triangle_points.restype=None
    lib.paired_triangle_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_void_p]
    lib.paired_triangle_points.restype=None
    return lib


def ellipsoid_projection(points,axes):
    lib=library()
    if lib is None:return None
    points=np.ascontiguousarray(points,dtype=float);axes=np.ascontiguousarray(axes,dtype=float)
    result=np.empty_like(points)
    lib.ellipsoid_points(len(points),points.ctypes.data,axes.ctypes.data,result.ctypes.data)
    return result


def triangle_projection(points,triangles):
    lib=library()
    if lib is None:return None
    from trimesh.constants import tol
    points=np.ascontiguousarray(points,dtype=float);triangles=np.ascontiguousarray(triangles,dtype=float)
    result=np.empty_like(points)
    lib.nearest_triangle_points(len(points),points.ctypes.data,len(triangles),triangles.ctypes.data,tol.zero,result.ctypes.data)
    return result


def project_to_triangles(triangles,points):
    import trimesh
    lib=library()
    if lib is None:return trimesh.triangles.closest_point(triangles,points)
    points=np.ascontiguousarray(points,dtype=float);triangles=np.ascontiguousarray(triangles,dtype=float)
    if triangles.shape!=(len(points),3,3) or points.shape!=(len(points),3):raise ValueError('paired triangle dimensions')
    result=np.empty_like(points)
    lib.paired_triangle_points(len(points),points.ctypes.data,triangles.ctypes.data,trimesh.constants.tol.zero,result.ctypes.data)
    return result


class GeometryKernel:
    def __init__(self,model,frames,bounds,obstacles,floor_clearance,nx):
        self.lib=library()
        if self.lib is None:raise RuntimeError('build geometry kernel before using it')
        ids=np.ascontiguousarray(frames,dtype=np.int32)
        bids=np.ascontiguousarray([b[0] for b in bounds],dtype=np.int32)
        centers=np.ascontiguousarray([b[1] for b in bounds],dtype=float)
        halves=np.ascontiguousarray([b[2] for b in bounds],dtype=float)
        boxes=np.ascontiguousarray(obstacles,dtype=float)
        self.handle=self.lib.geometry_create(model.saveToString().encode(),len(ids),ids.ctypes.data,
            len(bids),bids.ctypes.data,centers.ctypes.data,halves.ctypes.data,
            len(obstacles),boxes.ctypes.data,int(floor_clearance is not None),float(floor_clearance or 0.))
        if not self.handle:raise RuntimeError(self.lib.geometry_error().decode())
        self.poses=np.empty((len(frames),12));self.jacobians=np.empty((len(frames),6,nx))
        self.residual=np.empty(len(bounds)*(len(obstacles)+int(floor_clearance is not None)))
        self.derivative=np.empty((len(self.residual),nx));self.nx=nx

    def evaluate(self,q,lift):
        q=np.ascontiguousarray(q,dtype=float);lift=np.ascontiguousarray(lift,dtype=float)
        status=self.lib.geometry_evaluate(self.handle,q.ctypes.data,lift.ctypes.data,self.nx,
            self.poses.ctypes.data,self.jacobians.ctypes.data,self.residual.ctypes.data,self.derivative.ctypes.data)
        if status:raise RuntimeError(self.lib.geometry_error().decode())
        return self.poses,self.jacobians,self.residual,self.derivative

    def fit(self,q,lift,points,normals,shape,gap,delta):
        kind=0 if shape is None else {'box':1,'elongated_box':1,'cylinder':2,'ellipsoid':3,'triangular_prism':4,'tapered_box':4}.get(getattr(shape,'family',None))
        if kind is None:return None
        from trimesh.constants import tol
        equations=np.ascontiguousarray(shape.equations if kind==4 else [],dtype=float)
        triangles=np.ascontiguousarray(shape.triangles if kind==4 else [],dtype=float)
        arrays=[np.ascontiguousarray(a,dtype=float) for a in
                [q,lift,points,normals,np.ones(3) if shape is None else shape.size/2,delta]]
        q,lift,points,normals,size,delta=arrays
        rows=len(points)*(7 if kind else 6)+len(self.residual)+self.nx
        residual=np.empty(rows);derivative=np.empty((rows,self.nx))
        status=self.lib.geometry_fit_evaluate(self.handle,q.ctypes.data,lift.ctypes.data,self.nx,
            len(points),points.ctypes.data,normals.ctypes.data,kind,size.ctypes.data,float(gap),
            len(equations),equations.ctypes.data,len(triangles),triangles.ctypes.data,float(tol.zero),
            delta.ctypes.data,residual.ctypes.data,derivative.ctypes.data)
        if status:raise RuntimeError(self.lib.geometry_error().decode())
        return residual,derivative

    def __del__(self):
        if getattr(self,'handle',None):self.lib.geometry_destroy(self.handle);self.handle=None
