"""Partition a convex solid without simplifying its outer boundary."""
import numpy as np
from scipy.spatial import ConvexHull


def split_convex(vertices, vertex_limit=120):
    """Return convex parts under a vertex budget, using exact plane clipping.

    Unlike reducing a hull's vertex count, splitting retains the union of the
    original solid. The default also keeps triangular faces below PhysX's 255
    polygon limit. This is a part-geometry operation, not a structure catalogue.
    """
    vertices=np.asarray(vertices,dtype=float);hull=ConvexHull(vertices)
    vertices=vertices[hull.vertices];hull=ConvexHull(vertices)
    if len(vertices)<=vertex_limit:
        faces=hull.simplices.copy()
        triangles=vertices[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        reverse=np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0
        faces[reverse]=faces[reverse][:,::-1]
        return [(vertices,faces)]
    edges=np.unique(np.sort(np.concatenate([hull.simplices[:,[0,1]],hull.simplices[:,[1,2]],hull.simplices[:,[2,0]]]),axis=1),axis=0)
    candidates=[]
    # Cutting along a long extrusion's axis can keep the same vertex count.
    # Compare all three plane directions using geometry alone.
    for axis in range(3):
        level=float(np.median(vertices[:,axis]))
        if not vertices[:,axis].min()<level<vertices[:,axis].max():continue
        signed=vertices[:,axis]-level
        crossing=edges[signed[edges[:,0]]*signed[edges[:,1]]<0]
        first,second=crossing.T;fraction=signed[first]/(signed[first]-signed[second])
        intersection=vertices[first]+fraction[:,None]*(vertices[second]-vertices[first])
        intersection[:,axis]=level
        halves=[]
        for side in [signed<=0,signed>=0]:
            points=np.unique(np.vstack([vertices[side],intersection]),axis=0)
            halves.append(points[ConvexHull(points).vertices])
        candidates.append(halves)
    halves=min(candidates,key=lambda pair:max(map(len,pair)))
    if max(map(len,halves))>=len(vertices):raise ValueError('convex partition failed to reduce vertex count')
    return [piece for half in halves for piece in split_convex(half,vertex_limit)]
