"""Reject torque imbalance even when summed force alone can balance gravity."""
import numpy as np
from amsrr_det.equilibrium import object_wrench_obstruction


def contacts(points):
    return [dict(frame='pad'+str(i),target_frame='object',target_point=np.asarray(p),
                 normal=np.array([1. if i==0 else -1.,0.,0.]),mu=2.25,
                 max_force=30.,pad_force_limit=30.)for i,p in enumerate(points)]


def test_off_center_force_is_rejected_but_opposed_balanced_grasp_is_not():
    gravity=np.array([0.,0.,-9.80665])
    assert object_wrench_obstruction(contacts([[1,0,0],[-1,0,0]]),gravity,np.eye(3)) is None
    proof=object_wrench_obstruction(contacts([[1,0,0],[1,0,0]]),gravity,np.eye(3))
    assert proof and proof['residual_lower_bound']>1e-6


def test_rotation_preserves_the_object_wrench_check():
    R=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
    for points in ([[1,0,0],[-1,0,0]],[[1,0,0],[1,0,0]]):
        original=contacts(points);rotated=contacts(points)
        for c in rotated:c['normal']=R@c['normal']
        a=object_wrench_obstruction(original,np.array([0,0,-9.8]),np.eye(3))
        b=object_wrench_obstruction(rotated,R@np.array([0,0,-9.8]),R)
        assert (a is None)==(b is None)
