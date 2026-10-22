"""Queries of a known, closed material surface; no simulator state or forces."""
import numpy as np
import trimesh
import pinocchio as pin
import coal
from scipy.spatial import ConvexHull


class MeshSurface:
    def __init__(self, nominal, mesh, convexes=None, *, query_backend='coal'):
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('known material surface must be closed and oriented')
        self.nominal=nominal;self.mesh=mesh
        self.vertices=np.asarray(mesh.vertices);self.faces=np.asarray(mesh.faces)
        vertices=coal.StdVec_Vec3s();faces=coal.StdVec_Triangle()
        for v in self.vertices:vertices.append(v)
        for f in self.faces:faces.append(coal.Triangle(*map(int,f)))
        self.bvh=coal.BVHModelOBBRSS();self.bvh.beginModel(len(faces),len(vertices));self.bvh.addSubModel(vertices,faces);self.bvh.endModel()
        self.request=coal.DistanceRequest();self.origin=coal.Transform3s();self.point=coal.Sphere(0.)
        self.convexes=None if convexes is None else [(part.bounds,ConvexHull(part.vertices).equations) for part in convexes]
        if self.convexes is not None:self.bounds=np.array([bounds for bounds,_ in self.convexes])
        self.batch_query=None
        if query_backend=='warp':
            from .mesh_query import WarpNearest
            self.batch_query=WarpNearest(mesh)
        elif query_backend!='coal':raise ValueError('unknown mesh query backend')

    def __getattr__(self,name):
        return getattr(self.nominal,name)

    def lowest_z(self,pose):
        return float(pose[2]+(self.vertices@pin.Quaternion(np.asarray(pose)[3:]).matrix()[2]).min())

    def contains(self,points):
        if self.convexes is None:return self.mesh.contains(points)
        inside=np.zeros(len(points),dtype=bool)
        possible=np.all((points[:,None,:]>=self.bounds[None,:,0,:]-1e-10)&(points[:,None,:]<=self.bounds[None,:,1,:]+1e-10),axis=2)
        for j in np.flatnonzero(possible.any(axis=0)):
            eq=self.convexes[j][1];ids=np.flatnonzero(~inside&possible[:,j])
            if len(ids):inside[ids]=np.all(points[ids]@eq[:,:3].T+eq[:,3]<=1e-10,axis=1)
        return inside

    def project_surface(self,points):
        x=np.atleast_2d(np.asarray(points,dtype=float))
        p=np.empty_like(x);distance=np.empty(len(x));ids=np.empty(len(x),dtype=int)
        result=coal.DistanceResult();placement=coal.Transform3s()
        if self.batch_query is not None:p,distance,ids=self.batch_query(x)
        else:
            for i,point in enumerate(x):
                result.clear();placement.setTranslation(point)
                distance[i]=coal.distance(self.bvh,self.origin,self.point,placement,self.request,result)
                p[i]=result.getNearestPoint1();ids[i]=result.b1
        normals=self.mesh.face_normals[ids];offset=x-p
        projection=x-np.einsum('ij,ij->i',offset,normals)[:,None]*normals
        barycentric=trimesh.triangles.points_to_barycentric(self.mesh.triangles[ids],projection)
        interior=np.all((barycentric>=-1e-9)&(barycentric<=1+1e-9),axis=1)
        inside=np.einsum('ij,ij->i',offset,normals)<0
        # Edge/vertex projections need containment, because a single adjacent
        # face cannot determine the sign at a non-convex boundary.
        if np.any(~interior):inside[~interior]=self.contains(x[~interior])
        gap=np.where(inside,-distance,distance)
        outward=np.divide(offset,gap[:,None],out=normals.copy(),where=abs(gap[:,None])>1e-12)
        return p,gap,outward
