import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.rotor_distribution import rotor_actions, preserve_wrench_distribution, bound_joint_loads, preserve_requested_action


def test_reallocation_keeps_wrench_and_respects_thrust_bounds():
    # Equal total lift, but opposing joint actions. The unconstrained optimum
    # [2, 2] exceeds the second rotor's bound, so [3, 1] is the correct answer.
    base=np.array([[1.,1.]])
    force,info=preserve_wrench_distribution(base,np.array([[1.,-1.]]),
                                          np.array([4.,0.]),np.array([2.,2.]),
                                          np.array([5.,1.]),np.array([1.]))
    assert info['accepted']
    np.testing.assert_allclose(force,[3.,1.],atol=1e-7)
    np.testing.assert_allclose(base@force,[4.],atol=1e-8)


def test_no_alternative_wrench_preserving_distribution_keeps_baseline():
    baseline=np.array([2.,3.])
    force,info=preserve_wrench_distribution(np.eye(2),np.ones((1,2)),baseline,
                                          np.zeros(2),np.full(2,20.),np.ones(1))
    assert not info['accepted']
    np.testing.assert_array_equal(force,baseline)


def test_rotor_action_matrix_matches_independent_inverse_dynamics():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    for i,a in enumerate(meta['actuators']):q[a['q']]=.1*np.sin(i)
    force=np.arange(len(meta['rotors']),dtype=float)+1.
    action=rotor_actions(model,meta,q);data=model.createData()
    external=[pin.Force.Zero() for _ in range(model.njoints)]
    for rotor,value in zip(meta['rotors'],force):
        frame=model.frames[rotor['frame']]
        external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,value]),np.zeros(3)))
    zero=np.zeros(model.nv)
    without=pin.rnea(model,data,q,zero,zero).copy()
    with_force=pin.rnea(model,data,q,zero,zero,external).copy()
    np.testing.assert_allclose(action@force,without-with_force,atol=1e-10)


def test_joint_bound_changes_internal_loading_without_changing_lift():
    # Lift=4, torque=f2-f1. Closest feasible force to [4,0] at |torque|<=1
    # is [2.5,1.5], including duplicate base constraints for identical poses.
    force,info=bound_joint_loads(np.array([[1.,1.],[1.,1.]]),np.array([[1.,-1.]]),
        np.array([0.]),np.array([4.,0.]),np.full(2,5.),np.array([1.]))
    assert info['accepted']
    np.testing.assert_allclose(force,[2.5,1.5],atol=1e-8)
    assert abs(force.sum()-4)<1e-8


def test_incompatible_lift_rotor_and_joint_bounds_return_no_command():
    # Rotor 2 <=1 implies torque magnitude >=2 while preserving lift=4.
    force,info=bound_joint_loads(np.array([[1.,1.]]),np.array([[1.,-1.]]),
        np.array([0.]),np.array([4.,0.]),np.array([5.,1.]),np.array([1.]))
    assert force is None and not info['accepted']


def test_paired_action_preserves_unsaturated_request_and_joint_command_bounds():
    # Original joint request=2, rotor joint action=0. The position-command
    # upper bound is tighter than the torque limit: transfer 1.1 to rotors.
    base=np.array([[1.,1.]])
    joint=np.array([[1.,-1.]])
    baseline=np.array([2.,2.])
    command,force,info=preserve_requested_action(base,joint,baseline,np.array([.02]),
        np.zeros(1),np.zeros(1),np.array([100.]),np.array([5.]),np.ones(1),
        np.array([[-1.,.009]]),np.full(2,5.))
    assert info['accepted']
    np.testing.assert_allclose(force,[2.55,1.45],atol=1e-8)
    np.testing.assert_allclose(command,[.009],atol=1e-8)
    np.testing.assert_allclose(joint@force+100*command,[2.],atol=1e-8)
    np.testing.assert_allclose(base@force,base@baseline,atol=1e-8)


def test_paired_action_keeps_feasible_input_exactly_and_rejects_impossible_request():
    values=(np.array([[1.,1.]]),np.array([[1.,-1.]]),np.array([2.,2.]),np.array([.005]),
        np.zeros(1),np.zeros(1),np.array([100.]),np.array([5.]),np.ones(1),
        np.array([[-1.,1.]]),np.full(2,5.))
    command,force,info=preserve_requested_action(*values)
    assert info['accepted'] and info['unchanged']
    np.testing.assert_array_equal(command,values[3]);np.testing.assert_array_equal(force,values[2])
    values=list(values);values[0]=np.eye(2);values[3]=np.array([.02])
    command,force,info=preserve_requested_action(*values)
    assert command is None and force is None and not info['accepted']
