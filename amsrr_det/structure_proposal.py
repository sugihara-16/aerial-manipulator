"""Task-derived geometry from legal ports; local fit failures prove no exclusion.

Guided fits resolve free rotations against gravity and hold initial target
faces. Unguided fits permit arbitrary sampled surface directions and folds.
Material, force, finite-slip and approach checks remain the caller's job.
"""
import time
import numpy as np
import pinocchio as pin
from .model import build_model
from .structure_groups import sample_assignment
from .morphology import align_seed
from .grasp_geometry import fit_contacts


def proposal(group,shape,source,obstacles,rng, *, contact_count=None,guided=False,fixed_targets=False):
    started=time.perf_counter()
    edges,surfaces=sample_assignment(group,rng,contact_count=contact_count)
    m,meta=build_model(group['modules'],edges,shape.payload())
    q=pin.neutral(m);q[meta['object']['q']:]=source
    folded=not guided or rng.random()<.25
    for a in meta['actuators']:
        if a['role']=='dock' and folded:q[a['q']]=rng.uniform(a['lower']+.04,a['upper']-.04)
    directions=rng.normal(size=(len(surfaces),3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    if guided and len(surfaces)>=2 and rng.random()<.75:
        # Task-derived opposing surface directions are initial guesses only.
        # All surface counts and arbitrary directions retain positive support.
        axis=np.eye(3)[int(rng.integers(3))]
        directions[:2]=np.array([axis,-axis])
    samples=[shape.ray_surface(direction) for direction in directions]
    points=np.array([item[0] for item in samples]);normals=-np.array([item[1] for item in samples])
    R=pin.Quaternion(source[3:]).matrix();world=points@R.T+source[:3];wn=normals@R.T
    aligned,_=align_seed(m,meta,q,surfaces,world+.001*wn,wn,upright=guided)
    # A random rigid rotation retains variation even for a one-pad fit whose
    # alignment has an undetermined rotation about the surface normal.
    if len(surfaces)==1:
        angle=rng.uniform(-np.pi,np.pi);turn=pin.exp3(wn[0]*angle)
        center=world[0]+.001*wn[0]
        aligned[:3]=center+turn@(aligned[:3]-center)
        aligned[3:7]=pin.Quaternion(turn@pin.Quaternion(aligned[3:7]).matrix()).coeffs()
    closed,fit=fit_contacts(m,meta,aligned,surfaces,world+.001*wn,wn,object_shape=None if fixed_targets else shape,
                           floor_clearance=.03,base_rotation=True,obstacles=obstacles,max_evaluations=45)
    if fixed_targets:fit.update(object_points=points.tolist(),inward_normals=normals.tolist())
    return dict(group=group['id'],modules=group['modules'],edges=edges,surfaces=surfaces,
                q=closed.tolist(),fit=fit,wall_s=time.perf_counter()-started)


