"""Check equality elimination against independently known constrained optima."""
import numpy as np
import pytest
from build.contact_allocation_qp import solve


def test_contact_floor_changes_optimum_without_changing_whole_action():
    # x0+x1=1; without pad floor, minimum is (0.5,0.5,0).
    H=np.eye(3);g=np.zeros(3);E=np.array([[1.,1.,0.]])
    C=np.vstack([np.eye(3),[1.,0.,1.]])
    x,residual,gap=solve(H,g,E,np.array([1.]),C,np.array([-2.,-2.,-2.,1.]),np.full(4,2.))
    np.testing.assert_allclose(x,[2/3,1/3,1/3],atol=2e-6)
    assert residual<1e-7 and gap<1e-8


def test_unequal_inertias_preserve_original_units():
    H=np.diag([1e-6,1.,1e6]);g=np.array([-1e-6,-.2,-1e6])
    E=np.array([[1.,1.,0.]])
    x,residual,gap=solve(H,g,E,np.array([.5]),np.eye(3),np.full(3,-2.),np.full(3,2.))
    second=(.2-.5e-6)/(1+1e-6)
    np.testing.assert_allclose(x,[.5-second,second,1.],atol=1e-6)
    assert residual<1e-7 and gap<1e-8


def test_infeasible_contact_condition_is_not_dropped():
    with pytest.raises(ValueError):
        solve(np.eye(3),np.zeros(3),np.array([[1.,1.,0.]]),np.array([0.]),
              np.vstack([np.eye(3),[1.,1.,0.]]),np.array([-2.,-2.,-2.,1.]),np.full(4,2.))


def test_supported_four_module_numerical_regression():
    from pathlib import Path
    problem=dict(np.load(Path(__file__).parent/'fixtures/contact_allocation_qp.npz'))
    x,residual,gap=solve(**problem)
    assert residual<1e-7 and gap<1e-8


def test_zero_pressure_row_and_no_wrench_equality():
    # An open contact contributes 0 <= 0 <= 30. It must neither destroy the
    # bounded closest-input optimum nor add a badly scaled dual variable.
    C=np.vstack([np.eye(2),np.zeros(2)])
    x,residual,gap=solve(np.eye(2),np.array([-2.,.25]),np.zeros((0,2)),np.zeros(0),
                         C,np.array([-1.,-1.,0.]),np.array([1.,1.,30.]))
    np.testing.assert_allclose(x,[1.,-.25],atol=1e-6)
    assert residual<1e-7 and gap<1e-8


def test_zero_pressure_row_cannot_supply_positive_preload():
    with pytest.raises(ValueError,match='constant constraint infeasible'):
        solve(np.eye(2),np.zeros(2),np.zeros((0,2)),np.zeros(0),
              np.vstack([np.eye(2),np.zeros(2)]),np.array([-1.,-1.,.5]),np.array([1.,1.,30.]))
