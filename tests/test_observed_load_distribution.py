import numpy as np
from amsrr_det.rotor_distribution import redistribute_observed_load


def test_motor_burden_can_move_to_rotors_without_changing_total_action():
    base=np.array([[1.,1.]]);joint=np.array([[1.,-1.]])
    actual=np.array([1.3]);baseline=np.array([2.,2.])
    force,info=redistribute_observed_load(base,joint,actual,baseline,np.array([5.,5.]),np.array([1.3]))
    assert info['accepted']
    np.testing.assert_allclose(force,[2.65,1.35],atol=1e-6)
    np.testing.assert_allclose(base@(force-baseline),0.,atol=1e-10)
    np.testing.assert_allclose(joint@(force-baseline)+np.array(info['conditional_motor_target'])-actual,0.,atol=1e-10)


def test_unmodified_gimbal_action_must_be_preserved_too():
    preserved=np.array([[1.,1.],[1.,-1.]])
    force,info=redistribute_observed_load(preserved,np.array([[1.,-1.]]),np.array([1.3]),np.array([2.,2.]),np.array([5.,5.]),np.array([1.3]))
    assert force is None and info['reason']=='no_preserved_action_nullspace'


def test_limited_rotor_headroom_limits_the_attainable_motor_relief():
    force,info=redistribute_observed_load(np.array([[1.,1.]]),np.array([[1.,-1.]]),np.array([1.3]),np.array([2.,2.]),np.array([2.2,5.]),np.array([1.3]))
    assert info['accepted']
    np.testing.assert_allclose(force,[2.2,1.8],atol=1e-6)
    np.testing.assert_allclose(info['conditional_motor_target'],[.9],atol=1e-6)


def test_a_frozen_prediction_cannot_hide_an_unrelievable_other_motor():
    force,info=redistribute_observed_load(np.array([[1.,1.]]),np.array([[1.,-1.],[0.,0.]]),np.array([1.3,1.5]),np.array([2.,2.]),np.array([5.,5.]),np.array([1.3,1.3]))
    assert force is None and not info['accepted']
