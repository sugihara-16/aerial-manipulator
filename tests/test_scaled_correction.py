import numpy as np
import pytest
from build.scaled_correction import scaled_correction


def test_infeasible_request_reaches_largest_ray_fraction():
    x,a,_=scaled_correction(np.eye(2),np.zeros(2),np.zeros(2),np.ones((1,2)),np.array([3.]),np.zeros(2),np.ones(2))
    assert abs(a-2/3)<1e-10
    np.testing.assert_allclose(x,[1.,1.],atol=1e-8)


def test_feasible_request_preserves_original_minimum():
    x,a,_=scaled_correction(np.diag([1.,4.]),np.zeros(2),np.zeros(2),np.ones((1,2)),np.array([1.2]),np.zeros(2),np.ones(2))
    assert a==1.
    np.testing.assert_allclose(x,[.96,.24],atol=1e-8)


def test_zero_correction_need_not_be_feasible():
    x,a,_=scaled_correction(np.eye(1),np.zeros(1),np.zeros(1),np.ones((1,1)),np.array([2.]),np.array([.5]),np.ones(1))
    assert a==.5
    np.testing.assert_allclose(x,[1.],atol=1e-9)


def test_empty_ray_domain_is_rejected():
    with pytest.raises(ValueError,match='no attainable correction scale'):
        scaled_correction(np.eye(1),np.zeros(1),np.zeros(1),np.ones((1,1)),np.array([-2.]),np.array([.5]),np.ones(1))


def test_internal_objective_is_not_scaled_with_outer_correction():
    # x0 must hit 1 (alpha=.5). Only the feedback part of the linear term
    # scales: g1=-1 + .5*-.4=-1.2, so the free optimum is x1=1.2.
    x,a,_=scaled_correction(np.eye(2),np.array([0.,-1.4]),np.array([0.,-.4]),
        np.array([[1.,0.]]),np.array([2.]),np.array([0.,-2.]),np.array([1.,2.]))
    assert a==.5
    np.testing.assert_allclose(x,[1.,1.2],atol=1e-8)
