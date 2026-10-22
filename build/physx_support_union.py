"""Read-only checks of the cooked support via the public scene-query API.

The authored boundary has an exact cell proof. The cooked geometry is sampled
on every authored face and through every arrangement cell; it is not extracted
as a mesh and this diagnostic does not prove all unsampled cooked geometry.
"""
from itertools import product
import numpy as np
from omni.physx import get_physx_scene_query_interface


def validate_cooked_union(info):
    query=get_physx_scene_query_interface();path=info['path'];errors=[];faces=[];rays=0
    def cast(origin,direction,distance,both=False):
        hits=[]
        def report(hit):
            if hit.collision==path:hits.append((float(hit.distance),np.array(hit.normal),int(hit.face_index)))
            return True
        query.raycast_all(tuple(map(float,origin)),tuple(map(float,direction)),float(distance),report,both)
        return min(hits,key=lambda h:h[0]) if hits else None
    for point,normal in zip(info['face_centers'],info['face_normals']):
        point,normal=np.asarray(point),np.asarray(normal)
        hit=cast(point+.001*normal,-normal,.002);rays+=1
        if hit is None:raise ValueError('cooked support missing an authored outer face')
        errors.append(abs(hit[0]-.001));faces.append(hit[2])
        if np.dot(hit[1],normal)<1-1e-6:raise ValueError('cooked support outer normal mismatch')
    centers,half=np.asarray(info['centers']),np.asarray(info['half_sizes']);lo,hi=centers-half,centers+half
    planes=[np.unique(np.r_[lo[:,a],hi[:,a]]) for a in range(3)]
    middles=[(p[:-1]+p[1:])/2 for p in planes]
    inside=lambda p:np.any(np.all((p>lo)&(p<hi),axis=1))
    for point in map(np.asarray,product(*middles)):
        for axis in range(3):
            for sign in [-1,1]:
                direction=np.eye(3)[axis]*sign;expected=[]
                for boundary in planes[axis]:
                    distance=(boundary-point[axis])*sign
                    if distance<=1e-7:continue
                    on=point+distance*direction
                    if inside(on-direction*1e-7)!=inside(on+direction*1e-7):expected.append(distance)
                hit=cast(point,direction,10.,True);rays+=1
                if bool(expected)!=(hit is not None):raise ValueError('cooked support changed occupied boundary')
                if expected:errors.append(abs(hit[0]-min(expected)))
    maximum=max(errors)
    if maximum>1e-6:raise ValueError(f'cooked support ray error {maximum}')
    return dict(**{k:v for k,v in info.items() if k not in ['face_centers','face_normals']},
        qualification='authored_exact_boundary_and_cooked_scene_ray_samples',
        cooked_mesh_extracted=False,rays=rays,maximum_ray_error_m=maximum,
        hit_cooked_face_indices=len(set(faces)),ray_sample_check_passed=True)
