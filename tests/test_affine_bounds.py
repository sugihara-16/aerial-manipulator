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


def test_reused_bounds_match_independent_solves_across_directions_and_origins():
    from amsrr_det.affine_bounds import ScalarBounds,scalar_bound
    rng=np.random.default_rng(94)
    for n in (8,12,16):
        matrix=rng.normal(size=(6,n));lower=-rng.random(n);upper=rng.random(n)
        problem=ScalarBounds(matrix,lower,upper)
        for _ in range(16):
            base=matrix@rng.uniform(lower,upper);direction=rng.normal(size=6)
            expected=scalar_bound(matrix,base,direction,lower,upper,nonnegative=True)
            actual=problem.bound(base,direction)
            assert actual is not None and expected is not None
            np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-8)
        # Reusing a successful basis must not hide an infeasible slice.
        assert problem.bound(np.full(6,1e6),np.zeros(6)) is None


def test_contact_constraints_shrink_the_same_wrench_slice():
    # x0+x1 provides translation. Contact constraints consume part of each
    # actuator's capability even though the actuator box alone permits more.
    A=np.array([[1.,1.]])
    constraints=(np.array([[1.,0.],[0.,1.]]),np.array([.3,-np.inf]),np.array([.7,.2]))
    interval=scalar_interval(A,np.zeros(1),np.ones(1),[0.,0.],[1.,1.],linear_constraints=constraints)
    np.testing.assert_allclose(interval,[.3,.9],atol=1e-9)
    assert scalar_interval(A,np.zeros(1),np.ones(1),[0.,0.],[1.,1.],
        linear_constraints=(np.array([[1.,1.]]),np.array([3.]),np.array([np.inf]))) is None

def test_contact_slice_endpoints_match_independent_linear_program():
    rng=np.random.default_rng(417)
    for _ in range(12):
        A=rng.normal(size=(3,8));direction=rng.normal(size=3)
        extra=rng.normal(size=(6,8));lower=-rng.random(8);upper=rng.random(8)
        witness=rng.uniform(lower,upper);base=A@witness
        lo=extra@witness-rng.random(6);hi=extra@witness+rng.random(6)
        result=scalar_interval(A,base,direction,lower,upper,linear_constraints=(extra,lo,hi))
        assert result is not None
        for index,sign in enumerate([1.,-1.]):
            answer=linprog(np.r_[np.zeros(8),sign],A_eq=np.c_[A,-direction],b_eq=base,
                A_ub=np.c_[np.vstack([extra,-extra]),np.zeros(12)],b_ub=np.r_[hi,-lo],
                bounds=[*zip(lower,upper),(None,None)])
            assert answer.success
            assert abs(answer.x[-1]-result[index])<1e-7


def test_changing_interval_reuses_workspace_without_stale_bounds():
    from amsrr_det.affine_bounds import RepeatedScalarInterval
    problem=RepeatedScalarInterval();rng=np.random.default_rng(20261027)
    for n in [8,8,12,12]:
        for _ in range(8):
            matrix=rng.normal(size=(3,n));lo=-rng.random(n);hi=rng.random(n)
            base=matrix@rng.uniform(lo,hi);direction=rng.normal(size=3)
            np.testing.assert_allclose(problem(matrix,base,direction,lo,hi),
                scalar_interval(matrix,base,direction,lo,hi),rtol=1e-8,atol=1e-8)
        assert problem(matrix,np.full(3,1e6),np.zeros(3),lo,hi) is None
        assert problem(matrix,base,direction,lo,hi) is not None
