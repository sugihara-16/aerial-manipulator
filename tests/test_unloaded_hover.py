"""Physical residual checks remain mandatory after numerical refinement."""
from pathlib import Path
import numpy as np
import pytest
from build.unloaded_hover import _closest_input


def test_six_module_gravity_allocation_meets_original_physical_tolerance():
    problem=dict(np.load(Path(__file__).parent/'fixtures/unloaded_six_module.npz'))
    x,residual=_closest_input(**problem)
    assert residual<=1e-7
    assert np.max(abs(problem['A']@x-problem['rhs']))<=1e-7
    assert np.max(problem['lo']-x)<=1e-7
    assert np.max(x-problem['hi'])<=1e-7


def test_closest_input_retains_equilibrium_and_bound_active_optimum():
    x,_=_closest_input(np.array([[1.,1.]]),np.array([1.]),
                       np.zeros(2),np.ones(2),np.array([2.,-1.]))
    np.testing.assert_allclose(x,[1.,0.],atol=1e-7)


def test_refinement_does_not_accept_impossible_gravity_balance():
    with pytest.raises(ValueError,match='unloaded gravity allocation unresolved'):
        _closest_input(np.array([[1.,1.]]),np.array([3.]),
                       np.zeros(2),np.ones(2),np.zeros(2))
