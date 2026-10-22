"""Contact candidates on real material triangles and known object geometry.

A close pair is a geometric candidate, never an imposed sticking constraint or
an observed force. No simulation contacts are read by this module.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import trimesh
from .model import ROOT,transform
from .grasp_geometry import grasp_frame
from .native_grasp_geometry import project_to_triangles


def material_cache_identity(port,mesh_cache):
    """Only the authored one-pad geometry; no task or assembled structure."""
    source=ROOT/'assets/robots/holon/holon.urdf';raw=source.read_bytes()
    link=('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]
    mesh=ET.fromstring(raw).find(f"link[@name='{link}']/collision/geometry/mesh")
    mesh_bytes=(source.parent/mesh.get('filename')).read_bytes()
    digest=hashlib.sha256(raw+mesh_bytes+Path(__file__).read_bytes()+
        (ROOT/'amsrr_det/model.py').read_bytes()+trimesh.__version__.encode()+np.__version__.encode()+pin.__version__.encode())
    if mesh_cache is not None:
        directory=Path(mesh_cache)/hashlib.sha256(mesh_bytes).hexdigest()
        manifest=(directory/'manifest.json').read_bytes();digest.update(manifest)
        for part in json.loads(manifest)['parts']:digest.update((directory/part['file']).read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=12)
def material_triangles(port,mesh_cache=None,*,use_cache=True):
    if use_cache:
        cache=ROOT/'assets/generated/material_surfaces.npz'
        try:
            key=material_cache_identity(port,mesh_cache)
            with np.load(cache,allow_pickle=False) as saved:
                if key in saved:return saved[key]
        except (OSError,ValueError,KeyError):pass
    from .model import build_model
    model,meta=build_model()
    link=('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]
    source=ROOT/'assets/robots/holon/holon.urdf'
    collision=ET.parse(source).find(f"link[@name='{link}']/collision")
    mesh=collision.find('geometry/mesh');path=source.parent/mesh.get('filename')
    if mesh_cache is None:
        shape=trimesh.load_mesh(path)
    else:
        # These are known, offline collider surfaces, not simulator contacts or
        # contact-force observations. Keep the CAD form available for comparison.
        digest=hashlib.sha256(path.read_bytes()).hexdigest();directory=Path(mesh_cache)/digest
        manifest=json.loads((directory/'manifest.json').read_text());parts=[]
        if manifest['source_sha256']!=digest:raise ValueError('stale contact surface cache')
        for part in manifest['parts']:
            file=directory/part['file']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=part['sha256']:raise ValueError('contact surface hash mismatch')
            parts.append(trimesh.load_mesh(file))
        shape=trimesh.util.concatenate(parts)
    frame=meta['frames'][grasp_frame(0,port)];body=meta['frames'][f'm0/{link}']
    T=model.frames[frame].placement.inverse()*model.frames[body].placement*transform(collision.find('origin'))
    vertices=np.asarray(shape.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    vertices=vertices@T.rotation.T+T.translation;triangles=vertices[shape.faces]
    normals=np.asarray(shape.face_normals)@T.rotation.T
    # Include recessed material within the declared maximum penetration. The
    # front normal excludes the rear/connection side of a grasp mechanism.
    selected=(normals[:,0]>.3)&(triangles[:,:,0].max(axis=1)>vertices[:,0].max()-.003)
    return triangles[selected]


def close_material(shape,object_from_pad,triangles, *, band=.0002,max_points=12):
    """Refine nearest points inside each triangle, then retain the close patch.

    Alternating projections are a numerical search, not a global contact proof.
    Retained points lie on actual source triangles (including face interiors).
    Gaps and normals are evaluated against the curved object, not a fixed plane.
    """
    T=object_from_pad
    tri=triangles@T.rotation.T+T.translation
    centers=tri.mean(axis=1);radius=np.linalg.norm(tri-centers[:,None,:],axis=2).max(axis=1)
    _,gap,_=shape.project_surface(centers)
    # Signed Euclidean distance is 1-Lipschitz: a face farther than its bounding
    # radius cannot beat this already attained distance. Only refine survivors.
    tri=tri[gap-radius<=gap.min()+band]
    # Use the shape's quadratic metric as a second interior seed. Euclidean
    # projection toward a cylinder COM biases the result along its free axis;
    # alternating projection by a micrometre gap can then converge too slowly.
    points=project_to_triangles(tri,np.zeros((len(tri),3)))
    _,gaps,normals=shape.project_surface(points)
    weights=(2/shape.size if shape.family=='ellipsoid' else
             np.array([1.,1.,.001]) if shape.family=='cylinder' else None)
    if weights is not None:
        candidate=project_to_triangles(tri*weights,np.zeros((len(tri),3)))/weights
        _,trial,trial_normals=shape.project_surface(candidate);better=trial<gaps
        points[better]=candidate[better];gaps[better]=trial[better];normals[better]=trial_normals[better]
    steps=np.full((len(tri),1),.25*min(shape.size)**2/max(shape.size))
    for _ in range(8):
        candidate=project_to_triangles(tri,points-steps*normals)
        _,trial,trial_normals=shape.project_surface(candidate);better=trial<=gaps
        improvement=np.maximum(gaps-trial,0.)
        points[better]=candidate[better];gaps[better]=trial[better];normals[better]=trial_normals[better]
        steps[~better]*=.5
        if improvement.max(initial=0)<1e-10:break
    points=np.vstack([points,np.unique(tri.reshape(-1,3),axis=0)])
    surface,gap,normals=shape.project_surface(points)
    nearest=gap.min();eligible=np.flatnonzero(gap<=nearest+band)
    # Furthest point subsampling preserves patch extent without assigning a
    # separately free contact force to every CAD tessellation vertex.
    chosen=[int(eligible[np.argmin(gap[eligible])])]
    while len(chosen)<min(max_points,len(eligible)):
        distances=np.min(np.linalg.norm(points[eligible,None,:]-points[chosen][None,:,:],axis=2),axis=1)
        i=int(np.argmax(distances))
        if distances[i]<.001:break
        chosen.append(int(eligible[i]))
    indices=np.array(chosen)
    material=(points[indices]-T.translation)@T.rotation
    return material,surface[indices],gap[indices],normals[indices]
