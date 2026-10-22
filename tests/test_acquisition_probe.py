import numpy as np
import pinocchio as pin
from amsrr_det.grasp_task import probe_acquired, free_object_orientation_gain, probe_goal_height, object_height_at_clearance, observed_extent_rate, PROBE_HEIGHT
from amsrr_det.objects import ObjectShape


def test_probe_rejects_stationary_object_missing_pad_and_support_edge_pivot():
    shape=ObjectShape('box',[.30,.40,.20],.6);source=np.array([0,0,.30,0,0,0,1.])
    speed=np.zeros(6);contact=np.array([7e-6,20e-6,30e-6]);lifted=source.copy();lifted[2]+=.005
    assert probe_acquired(shape,source,lifted,speed,.20,contact)
    assert not probe_acquired(shape,source,source,speed,.20,contact)
    assert not probe_acquired(shape,source,lifted,speed,.20,np.array([-1e-3,20e-6,30e-6]))
    pivoted=source.copy();pivoted[3:]=pin.Quaternion(pin.rpy.rpyToMatrix(.02,0,0)).coeffs()
    pivoted[2]=.20-shape.lowest_z(np.r_[0.,0.,0.,pivoted[3:]])
    assert pivoted[2]>source[2]+.003
    assert not probe_acquired(shape,source,pivoted,speed,.20,contact)
    pivoted[2]+=.005
    assert probe_acquired(shape,source,pivoted,speed,.20,contact)


def test_proximity_permits_probe_but_does_not_replace_observed_lift():
    shape=ObjectShape('box',[.30,.40,.20],.6);source=np.array([0,0,.30,0,0,0,1.]);lifted=source.copy();lifted[2]+=.005
    near=np.array([-50e-6,-60e-6,50e-6]);speed=np.zeros(6)
    assert not probe_acquired(shape,source,lifted,speed,.20,near)
    assert probe_acquired(shape,source,lifted,speed,.20,near,near_gap_m=.0002)
    assert not probe_acquired(shape,source,source,speed,.20,near,near_gap_m=.0002)


def test_free_probe_can_correct_a_pose_that_its_acceptance_rejects():
    shape=ObjectShape('box',[.30,.40,.20],.6);source=np.array([0,0,.30,0,0,0,1.])
    pose=source.copy();pose[2]+=.005
    pose[3:]=pin.Quaternion(pin.rpy.rpyToMatrix(0,0,.067)).coeffs()
    assert not probe_acquired(shape,source,pose,np.zeros(6),.20,np.full(3,.00015))
    assert free_object_orientation_gain(shape.lowest_z(pose)-.20)>.99
    assert free_object_orientation_gain(0.)==0
    assert free_object_orientation_gain(-.001)==0


def test_probe_target_frees_tilted_corners_without_accepting_the_tilt():
    # Independent box vertices expose the old radius*0.05 clipping deadlock.
    from itertools import product
    shape=ObjectShape('box',[.30,.40,.20],.6)
    source=np.array([0.,0.,.30,0.,0.,0.,1.]);pose=source.copy()
    rotation=pin.rpy.rpyToMatrix(.12,.18,0.)
    pose[3:]=pin.Quaternion(rotation).coeffs()
    vertices=np.array(list(product([-.15,.15],[-.20,.20],[-.10,.10])))@rotation.T
    pose[2]=probe_goal_height(shape,source,pose,.20,3.)
    assert abs(np.min(vertices[:,2])+pose[2]-.20-PROBE_HEIGHT)<1e-12
    assert abs(np.min(vertices[:,2])+probe_goal_height(shape,source,pose,.20,0.)-.20)<1e-12
    assert probe_goal_height(shape,source,source,.20,0.)==source[2]
    assert free_object_orientation_gain(np.min(vertices[:,2])+pose[2]-.20)>.99
    assert not probe_acquired(shape,source,pose,np.zeros(6),.20,np.full(2,.00015))
    carrying_height=object_height_at_clearance(shape,pose[3:],.20,.12)
    assert abs(np.min(vertices[:,2])+carrying_height-.20-.12)<1e-12


def test_rotating_bottom_reference_matches_independent_vertices():
    from itertools import product
    shape=ObjectShape('box',[.30,.40,.20],.6)
    vertices=np.array(list(product([-.15,.15],[-.20,.20],[-.10,.10])))
    source=np.array([0.,0.,.30,0.,0.,0.,1.]);period=.01
    # Include a tied bottom face and a change of the lowest corner.
    for angles,omega in [([0.,0.,0.],[.03,-.05,.01]),
                         ([.12,.18,0.],[.03,-.05,.01]),
                         ([-.0001,0.,0.],[.03,0.,0.])]:
        rotation=pin.rpy.rpyToMatrix(*angles)
        pose=source.copy();pose[3:]=pin.Quaternion(rotation).coeffs()
        rate=observed_extent_rate(shape,pose,rotation.T@omega,period)
        z=probe_goal_height(shape,source,pose,.20,3.)
        future_rotation=pin.exp3(period*np.array(omega))@rotation
        bottom_now=(vertices@rotation.T)[:,2].min()+z-.20
        bottom_next=(vertices@future_rotation.T)[:,2].min()+z+period*rate-.20
        assert abs(bottom_now-PROBE_HEIGHT)<1e-12
        assert abs(bottom_next-bottom_now)<1e-12
        assert observed_extent_rate(shape,pose,np.zeros(3),period)==0.
