"""Constraint units must not erase a physically required input correction."""
import clarabel
import numpy as np
from build.held_input_solver import normalize_constraints, solve, violation


def test_tiny_distance_and_force_cone_keep_the_same_known_optimum():
    # min .5||x||² - 2*y, subject to x >= 2 and ||(y,z)|| <= 1.
    # Express the same constraints in metres and a separate force unit.
    cones=[clarabel.NonnegativeConeT(1),clarabel.SecondOrderConeT(3)]
    base=np.array([[-1.,0.,0.],[0.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    bound=np.array([-2.,1.,0.,0.]);c=np.array([0.,-2.,0.])
    for distance_scale,force_scale in [(1.,1.),(1e-12,1e4),(1e4,1e-12)]:
        units=np.array([distance_scale]+[force_scale]*3)
        matrix,rhs=normalize_constraints(base*units[:,None],bound*units,cones)
        x,_=solve(np.eye(3),c,matrix,rhs,cones)
        np.testing.assert_allclose(x,[2.,1.,0.],atol=2e-6)
        assert violation(base,bound,cones,x)<2e-6


def test_normalization_keeps_equalities_and_does_not_mutate_callers():
    matrix=np.array([[1e-12,2e-12],[0.,-1e6]])
    bound=np.array([5e-12,-2e6]);original=matrix.copy();rhs=bound.copy()
    cones=[clarabel.ZeroConeT(1),clarabel.NonnegativeConeT(1)]
    a,b=normalize_constraints(matrix,bound,cones)
    np.testing.assert_array_equal(matrix,original);np.testing.assert_array_equal(bound,rhs)
    assert violation(a,b,cones,np.array([1.,2.]))<1e-14
    assert violation(a,b,cones,np.array([0.,2.]))>.1


def test_requested_precision_keeps_a_small_required_correction():
    matrix=np.array([[-1.]])
    bound=np.array([-4e-9]);cones=[clarabel.NonnegativeConeT(1)]
    x,note=solve(np.eye(1),np.zeros(1),matrix,bound,cones,tolerance=1e-12)
    np.testing.assert_allclose(x,[4e-9],rtol=0.,atol=1e-12)
    assert note['maximum_violation']<=1e-12
