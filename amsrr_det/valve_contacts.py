"""Valve torque contacts are confined to the bar's two local y faces."""
import numpy as np
import pinocchio as pin
from .grasp_task import material_force_contacts
from .task_geometry import box_face_mask


def side_normals(normals):
    """The local z axis is the hinge axis, regardless of world mounting."""
    return np.abs(np.asarray(normals)[...,1])>=1.-1e-8


def require_side_contacts(plant,shape,q,surfaces):
    """Validate actual CAD material patches, including near-edge contacts."""
    contacts=material_force_contacts(plant,shape,q,surfaces,force_limit=None,material_tangents=True)
    data=plant.pin.createData();pin.framesForwardKinematics(plant.pin,data,q)
    rotation=data.oMf[plant.meta['object']['frame']].rotation
    if not contacts or not np.all(side_normals(np.array([c['normal'] for c in contacts])@rotation)):
        raise ValueError('valve contact material must stay on bar side faces')
    return contacts


def require_side_geometry(model,data,shape,*,ignore_geoms=()):
    """Include every real triangle, not only the authored anchor patch."""
    obj=model.geom('object').id
    rotation=data.geom_xmat[obj].reshape(3,3)
    contacts=data.contact;geoms=contacts.geom
    selected=((geoms[:,0]==obj)|(geoms[:,1]==obj))&(contacts.dist<=.0001)
    if ignore_geoms:selected&=~np.isin(geoms,list(ignore_geoms)).any(axis=1)
    if not selected.any():return
    normal=contacts.frame[selected,:3]*np.where(geoms[selected,1]==obj,1.,-1.)[:,None]
    distances=contacts.dist[selected]
    points=(contacts.pos[selected]+.5*distances[:,None]*normal-data.geom_xpos[obj])@rotation
    if not box_face_mask(points,shape.size,1,normal@rotation,distances).all():
        raise ValueError('rotation geometry touches a non-side bar face')
