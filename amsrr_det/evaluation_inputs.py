"""Construct resting object poses for simulation input, before any robot plan.

This descends the height of the COM supported by a horizontal plane. It does
not change the object's geometry, mass, or task dynamics. Already stable box
faces/cylinder end caps stay in their starting basin; an ellipsoid cannot be
specified with its unstable long axis upright as a released-placement target.
"""
import numpy as np
import pinocchio as pin


def resting_pose(shape,seed_pose):
    pose=np.asarray(seed_pose).copy();pose[:3]=0.
    rotation=pin.Quaternion(pose[3:]).matrix();step=.05;evaluations=0
    directions=np.array([[np.cos(a),np.sin(a),0.] for a in np.linspace(0,2*np.pi,8,endpoint=False)])
    def height(R):
        nonlocal evaluations
        evaluations+=1
        return -shape.lowest_z(np.r_[np.zeros(3),pin.Quaternion(R).coeffs()])
    current=height(rotation)
    while step>1e-7 and evaluations<4000:
        candidates=[pin.exp3(step*axis)@rotation for axis in directions]
        heights=[height(R) for R in candidates];best=int(np.argmin(heights))
        if heights[best]<current-1e-13:rotation=candidates[best];current=heights[best]
        else:step*=.5
    pose[3:]=pin.Quaternion(rotation).coeffs();pose[2]=height(rotation)
    return pose,dict(method='local_supported_COM_height_descent',evaluations=evaluations,
                     final_tilt_step_rad=step,initial_rotation=np.asarray(seed_pose)[3:].tolist())
