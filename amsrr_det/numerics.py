"""Explicit local error controls for experiments, not task safety margins."""

import pinocchio as pin
import numpy as np


def quintic_reference(model, meta, first, last, elapsed, duration):
    """Path position and derivatives at one declared time.

    Base velocity/acceleration are in world axes, as expected by flight
    controllers; joint derivatives retain the model's scalar coordinates.
    """
    if duration <= 0:
        raise ValueError('positive duration required')
    u = float(np.clip(elapsed / duration, 0., 1.))
    fraction = u**3 * (10 - 15*u + 6*u*u)
    delta = pin.difference(model, first, last)
    q = pin.interpolate(model, first, last, fraction)
    v = delta * (30*u*u*(1-u)**2 / duration)
    a = delta * (60*u*(1-u)*(1-2*u) / duration**2)
    world_v, world_a = v.copy(), a.copy()
    for base in meta['bases']:
        k, j = base['q'], base['v']
        rotation = pin.Quaternion(q[k+3:k+7]).matrix()
        world_v[j:j+3] = rotation @ v[j:j+3]
        world_v[j+3:j+6] = rotation @ v[j+3:j+6]
        world_a[j:j+3] = rotation @ (a[j:j+3] + np.cross(v[j+3:j+6], v[j:j+3]))
        world_a[j+3:j+6] = rotation @ a[j+3:j+6]
    return q, world_v, world_a, fraction


def advance_world_orientation(xyzw, rotation_vector):
    """Apply a world-axis rotation without accumulating quaternion norm error."""
    orientation=pin.Quaternion(xyzw);orientation.normalize()
    result=pin.Quaternion(pin.exp3(rotation_vector))*orientation
    result.normalize()
    return result.coeffs()


def diagnostic_tolerances(scale=1.):
    if scale<=0:raise ValueError('positive scale required')
    return {key:value*scale for key,value in dict(position_m=1e-6,angle_rad=1e-6,
        linear_velocity_m_s=1e-4,angular_velocity_rad_s=1e-4,joint_velocity_rad_s=1e-4,
        thrust_n=1e-3,torque_nm=1e-4,history_m=1e-6).items()}
