"""Evaluation-only resultants of the actual per-body external force arrays."""
import numpy as np
from scipy.spatial.transform import Rotation


def world_wrench(poses, forces, moments, origin):
    rotations=Rotation.from_quat(np.asarray(poses)[:,3:]).as_matrix()
    force=np.einsum('nij,nj->ni',rotations,forces)
    moment=np.einsum('nij,nj->ni',rotations,moments)
    return np.r_[force.sum(axis=0),(moment+np.cross(np.asarray(poses)[:,:3]-origin,force)).sum(axis=0)]
