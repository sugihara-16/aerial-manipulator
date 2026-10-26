"""Cheap sufficient overlap test, before constructing a full simulation model.

Every sample belongs to an authored convex collision piece. A sample strictly
inside the object proves overlap; no overlap found proves nothing. The complete
clearance checks remain mandatory for all surviving candidates.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pinocchio as pin
from .mujoco_model import _part_templates, CONTACT_CACHE


@lru_cache(maxsize=1)
def part_samples():
    assets,geometries,_=_part_templates(str(CONTACT_CACHE.resolve()))
    meshes={}
    for asset in assets:
        vertices=np.array([[float(x) for x in line.split()[1:4]]
            for line in Path(asset.get('file')).read_text().splitlines() if line.startswith('v ')])
        vertices*=np.fromstring(asset.get('scale','1 1 1'),sep=' ')
        ids=np.unique(np.r_[vertices.argmin(axis=0),vertices.argmax(axis=0)])
        meshes[asset.get('name')]=np.vstack([vertices[ids],vertices.mean(axis=0)])
    result=[]
    for link,geoms in geometries:
        points=[]
        for geom in geoms:
            p=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
            q=np.fromstring(geom.get('quat','1 0 0 0'),sep=' ')
            R=pin.Quaternion(np.r_[q[1:],q[0]]).matrix()
            points.append(meshes[geom.get('mesh')]@R.T+p)
        result.append((link,np.vstack(points)))
    return result


def object_overlap(model,meta,q,shape,source,surfaces):
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    allowed={f'm{i}/'+('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]
             for i,port in surfaces}
    R=pin.Quaternion(np.asarray(source)[3:]).matrix();a=shape.size/2
    for module in range(meta['modules']):
        for link,points in part_samples():
            name=f'm{module}/{link}'
            if name in allowed:continue
            T=data.oMf[meta['frames'][name]]
            local=((points@T.rotation.T+T.translation)-source[:3])@R
            # Eroded analytic interiors avoid classifying boundary roundoff.
            eps=1e-5
            if shape.family in ('box','elongated_box'):
                inside=np.all(abs(local)<a-eps,axis=1)
            elif shape.family=='ellipsoid':
                inside=np.sum((local/(a-eps))**2,axis=1)<1.
            elif shape.family=='cylinder':
                inside=(np.linalg.norm(local[:,:2],axis=1)<a[0]-eps)&(abs(local[:,2])<a[2]-eps)
            else:
                inside=np.all(local@shape.equations[:,:3].T+shape.equations[:,3]<-eps,axis=1)
            if np.any(inside):return name
    return None
