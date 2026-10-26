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
        ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,
        ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_void_p]
    lib.geometry_fit_evaluate.restype=ct.c_int
    lib.ellipsoid_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
    lib.ellipsoid_points.restype=None
    lib.nearest_triangle_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_void_p]
    lib.nearest_triangle_points.restype=None
    lib.paired_triangle_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_void_p]
    lib.paired_triangle_points.restype=None
    lib.refine_material_points.argtypes=[ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_void_p]
    lib.refine_material_points.restype=None
    lib.geometry_object_avoidance.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_void_p,ct.c_void_p]
    lib.geometry_object_avoidance.restype=ct.c_int
    lib.geometry_object_avoidance_grouped.argtypes=lib.geometry_object_avoidance.argtypes[:5]+[ct.c_void_p]+lib.geometry_object_avoidance.argtypes[5:]
    lib.geometry_object_avoidance_grouped.restype=ct.c_int
    lib.growth_clearance.argtypes=[ct.c_int]*6+[ct.c_void_p]*7
    lib.growth_clearance.restype=None
    return lib


def growth_clearance(base,boxes,incoming,pad,moving,rotation,source,size,obstacles):
    lib=library()
    if lib is None:return None
    k=boxes.shape[1];parts=boxes.shape[2]
    arrays=[np.ascontiguousarray(x,dtype=float) for x in
            (base,boxes,rotation,source,size,np.asarray(obstacles).reshape(-1,2,3))]
    result=np.empty((k*k,k))
    lib.growth_clearance(k,parts,incoming,pad,moving,len(arrays[-1]),
                        *[x.ctypes.data for x in arrays],result.ctypes.data)
    return result


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


def refine_material_points(shape,triangles):
    lib=library()
    if lib is None:return None
    from trimesh.constants import tol
    kind={'box':1,'elongated_box':1,'cylinder':2,'ellipsoid':3,'triangular_prism':4,'tapered_box':4}[shape.family]
    triangles=np.ascontiguousarray(triangles,dtype=float);axes=np.ascontiguousarray(shape.size/2)
    equations=np.ascontiguousarray(shape.equations if kind==4 else [],dtype=float)
    surface=np.ascontiguousarray(shape.triangles if kind==4 else [],dtype=float)
    result=np.empty((len(triangles),3))
    lib.refine_material_points(len(triangles),triangles.ctypes.data,kind,axes.ctypes.data,
        len(equations),equations.ctypes.data,len(surface),surface.ctypes.data,tol.zero,result.ctypes.data)
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

    def fit(self,q,lift,points,normals,shape,gap,delta,opposed_friction=None):
        kind=0 if shape is None else {'box':1,'elongated_box':1,'cylinder':2,'ellipsoid':3,'triangular_prism':4,'tapered_box':4}.get(getattr(shape,'family',None))
        if kind is None:return None
        from trimesh.constants import tol
        equations=np.ascontiguousarray(shape.equations if kind==4 else [],dtype=float)
        triangles=np.ascontiguousarray(shape.triangles if kind==4 else [],dtype=float)
        arrays=[np.ascontiguousarray(a,dtype=float) for a in
                [q,lift,points,normals,np.ones(3) if shape is None else shape.size/2,delta]]
        q,lift,points,normals,size,delta=arrays
        gaps=np.ascontiguousarray(np.broadcast_to(gap,len(points)),dtype=float)
        rows=len(points)*(7 if kind else 6)+len(self.residual)+self.nx+(3 if opposed_friction is not None else 0)
        residual=np.empty(rows);derivative=np.empty((rows,self.nx))
        status=self.lib.geometry_fit_evaluate(self.handle,q.ctypes.data,lift.ctypes.data,self.nx,
            len(points),points.ctypes.data,normals.ctypes.data,kind,size.ctypes.data,gaps.ctypes.data,
            len(equations),equations.ctypes.data,len(triangles),triangles.ctypes.data,float(tol.zero),float(opposed_friction or 0.),
            delta.ctypes.data,residual.ctypes.data,derivative.ctypes.data)
        if status:raise RuntimeError(self.lib.geometry_error().decode())
        return residual,derivative

    def object_avoidance(self,lift,object_frame,samples,shape,*,grouped=False):
        from trimesh.constants import tol
        kind={'box':1,'elongated_box':1,'cylinder':2,'ellipsoid':3,'triangular_prism':4,'tapered_box':4}[shape.family]
        frames,points=samples
        arrays=[np.ascontiguousarray(a,dtype=float) for a in (lift,points,shape.size/2,
            shape.equations if kind==4 else [],shape.triangles if kind==4 else [])]
        lift,points,size,equations,triangles=arrays
        frames=np.ascontiguousarray(frames,dtype=np.int32)
        starts=np.ascontiguousarray(np.r_[0,1+np.flatnonzero(np.diff(frames)),len(frames)],dtype=np.int32)
        count=len(starts)-1 if grouped else len(frames)
        r=np.empty(count);D=np.empty((count,self.nx))
        fn=self.lib.geometry_object_avoidance_grouped if grouped else self.lib.geometry_object_avoidance
        prefix=[starts.ctypes.data] if grouped else []
        status=fn(self.handle,lift.ctypes.data,self.nx,object_frame,count,*prefix,frames.ctypes.data,points.ctypes.data,kind,size.ctypes.data,len(equations),equations.ctypes.data,len(triangles),triangles.ctypes.data,float(tol.zero),r.ctypes.data,D.ctypes.data)
        if status:raise RuntimeError(self.lib.geometry_error().decode())
        return r,D

    def __del__(self):
        if getattr(self,'handle',None):self.lib.geometry_destroy(self.handle);self.handle=None
