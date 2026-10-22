import numpy as np
from scipy.optimize import linprog
from amsrr_det.affine_bounds import scalar_interval,l1_image_radius


def test_joint_axis_demand_limits_the_scalar_slice():
    # Independent x/y capabilities do not permit arbitrary simultaneous demand.
    A=np.array([[1.,1.],[1.,-1.]])
    interval=scalar_interval(A,np.array([1.,0.]),np.array([0.,1.]),[0.,0.],[1.,1.])
    np.testing.assert_allclose(interval,[-1.,1.],atol=1e-10)
    assert scalar_interval(A,np.array([3.,0.]),np.array([0.,1.]),[0.,0.],[1.,1.]) is None


def test_cross_polytope_radius_covers_convex_combinations_not_a_box():
    A=np.array([[1.,1.],[1.,-1.]])
    centers=[np.array([.2,0.]),np.array([-.2,0.])]
    F=np.eye(2)
    rho=l1_image_radius(A,centers,F,[-1.,-1.],[1.,1.])
    assert abs(rho-1.8)<1e-10
    rng=np.random.default_rng(8)
    for _ in range(20):
        z=rng.normal(size=2);z*=rho*rng.uniform()/np.sum(abs(z))
        center=centers[0]*rng.uniform();target=center+F@z
        feasible=linprog(np.zeros(2),A_eq=A,b_eq=target,bounds=[(-1.,1.)]*2)
        assert feasible.success
    # Simultaneous positive endpoints lie outside the same diamond.
    impossible=linprog(np.zeros(2),A_eq=A,b_eq=np.full(2,rho),bounds=[(-1.,1.)]*2)
    assert not impossible.success
