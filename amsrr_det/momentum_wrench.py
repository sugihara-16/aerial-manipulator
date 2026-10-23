"""Known external action in world axes about a specified moving origin.

Internal motors and robot/payload contact cancel for the complete assembly.
Environment reactions are deliberately absent: they are not observations
available to the controller.
"""
import numpy as np


def gravity_wrench(model, data, origin):
    force = model.body_mass[:, None] * model.opt.gravity
    return np.r_[force.sum(axis=0), np.cross(data.xipos - origin, force).sum(axis=0)]


def moving_origin_term(origin_velocity, linear_momentum):
    """Hdot = external_wrench - this term, with world-axis moments."""
    return np.r_[np.zeros(3), np.cross(origin_velocity, linear_momentum)]
