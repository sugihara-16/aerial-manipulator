"""Optional, task-derived contact refinement on a generated connection tree.

The soft objective reduces sensitivity of yaw moment to normal-load imbalance
on sampled material points. It does not certify force realization or release.
"""
import time
import numpy as np
import pinocchio as pin
from .mujoco_model import make_plant
from .grasp_geometry import fit_contacts,grasp_frame
from .grasp_task import refine_material_posture,finish_plan,support_obstacles
from .contact_geometry import close_material,material_triangles


def refine_normal_moments(old):
    """Return a checked new candidate or None; never modify the original plan."""
    start=time.perf_counter();shape=old.shape;meta=old.plant.meta
    plant=make_plant(meta['modules'],meta['edges'],shape.payload(),timestep=.00025,
                     mesh_cache=meta['material_mesh_cache'],floor=True)
    frames=[plant.meta['frames'][grasp_frame(*surface)] for surface in old.surfaces]
    triangles=[material_triangles(port,meta['material_mesh_cache']) for _,port in old.surfaces]
    data=plant.pin.createData();axis=np.array([0.,0.,1.])
    def patches(configuration):
        pin.framesForwardKinematics(plant.pin,data,configuration)
        obj=data.oMf[plant.meta['object']['frame']];samples=[];arms=[]
        for frame,mesh in zip(frames,triangles):
            transform=data.oMf[frame]
            material,_,gap,normal=close_material(shape,obj.inverse()*transform,mesh,band=.002,max_points=48)
            arm=np.cross(material@transform.rotation.T+transform.translation-obj.translation,normal@obj.rotation.T)@axis
            samples.append(material)
            arms.append(dict(sample_count=len(material),outward_normal_z_arm_range_m=[float(arm.min()),float(arm.max())],
                maximum_abs_z_arm_m=float(abs(arm).max()),minimum_material_gap_m=float(gap.min())))
        return samples,arms
    samples,before=patches(old.closed_q)
    rotation=pin.Quaternion(old.source[3:]).matrix()
    obstacles=support_obstacles(shape,old.source,old.destination,old.support_top)
    report=dict(static_candidate=False,before=before,
        scope='sampled normal-load yaw-moment sensitivity; static/geometric screens only',
        material_sampling='known front material, 2 mm band, at most 48 points per pad; not a continuous-region proof')
    candidate=None
    try:
        closed,fit=fit_contacts(plant.pin,plant.meta,old.closed_q,old.surfaces,
            old.points@rotation.T+old.source[:3],old.normals@rotation.T,
            object_shape=shape,base_rotation=True,floor_clearance=.03,obstacles=obstacles,max_evaluations=100,
            normal_moment_axis=axis,normal_moment_samples=samples)
        report['coarse_fit']=fit.copy()
        points=np.array(fit.pop('object_points'));normals=np.array(fit.pop('inward_normals'))
        opened,closed,fit=refine_material_posture(plant,shape,old.source,closed,old.surfaces,points,normals,obstacles=obstacles)
        _,report['after_material_refinement']=patches(closed)
        candidate=finish_plan(plant,shape,old.source,old.destination,old.support_top,
            opened,closed,old.surfaces,points,normals,fit,start)
        report['static_candidate']=True
    except ValueError as error:
        report.update(reason=str(error),evidence=getattr(error,'evidence',None))
    report['wall_s']=time.perf_counter()-start
    return candidate,report
