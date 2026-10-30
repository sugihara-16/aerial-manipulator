"""Independent optimality and infeasibility checks for the dense pose solve."""
import numpy as np
import pytest
from build.finite_slip_preload import _dense_pose


def test_dense_pose_matches_constructed_primal_dual_optimum():
    # Choose an optimum and active multipliers first, then construct g by KKT.
    rng=np.random.default_rng(20261023)
    R=rng.normal(size=(7,7));H=R.T@R+np.eye(7)*.01
    x=rng.normal(size=7);A=rng.normal(size=(12,7))
    slack=np.r_[np.zeros(3),rng.uniform(.2,2.,9)]
    b=A@x+slack;dual=np.r_[.2,.7,1.2,np.zeros(9)]
    g=-H@x-A.T@dual
    np.testing.assert_allclose(_dense_pose(H,g,A,b),x,atol=1e-10)


def test_dense_pose_rejects_infeasible_constraints():
    with pytest.raises(ValueError):
        _dense_pose(np.eye(1),np.zeros(1),np.array([[1.],[-1.]]),np.array([0.,-1.]))


def test_dependent_pose_rows_match_independent_conic_solver():
    # A five-module physical problem where whitened active constraints became
    # numerically inconsistent. Compare with a different convex solver.
    from pathlib import Path
    import scipy.sparse as sp
    import clarabel
    p=np.load(Path(__file__).with_name('fixtures')/'preload_dependent_rows.npz')
    H,g,A,b=[p[key] for key in ['H','g','A','b']]
    options=clarabel.DefaultSettings();options.verbose=False;options.max_threads=1
    options.tol_feas=options.tol_gap_abs=options.tol_gap_rel=1e-11
    other=clarabel.DefaultSolver(sp.csc_matrix(np.triu(H)),g,sp.csc_matrix(A),b,
        [clarabel.NonnegativeConeT(len(b))],options).solve()
    assert str(other.status)=='Solved'
    x=_dense_pose(H,g,A,b);reference=np.asarray(other.x)
    assert np.max(A@x-b)<1e-8
    assert abs((.5*x@H@x+g@x)-(.5*reference@H@reference+g@reference))<1e-6


def test_large_preload_active_equations_retain_original_feasibility_and_cost():
    # Captured from a five-module, 120 cm task. The unrefined active-set
    # solution passed feasibility but missed the objective precision contract.
    from pathlib import Path
    import scipy.sparse as sp
    import clarabel
    p=np.load(Path(__file__).with_name('fixtures')/'preload_active_roundoff.npz')
    H,g,A,b=[p[key] for key in ['H','g','A','b']]
    options=clarabel.DefaultSettings();options.verbose=False;options.max_threads=1
    options.tol_feas=options.tol_gap_abs=options.tol_gap_rel=1e-11
    reference=clarabel.DefaultSolver(sp.csc_matrix(np.triu(H)),g,sp.csc_matrix(A),b,
        [clarabel.NonnegativeConeT(len(b))],options).solve()
    assert str(reference.status)=='Solved'
    other=np.asarray(reference.x)
    x=_dense_pose(H,g,A,b)
    assert np.max(A@x-b)<1e-8
    assert abs((.5*x@H@x+g@x)-(.5*other@H@other+g@other))<1e-5


def test_identical_direct_input_problem_reuses_answer_but_changed_rhs_does_not(monkeypatch):
    import build.minimum_input_lp as module
    factory=module.clarabel.DefaultSolver;calls=[]
    def counted(*args,**kwargs):
        calls.append(True);return factory(*args,**kwargs)
    monkeypatch.setattr(module.clarabel,'DefaultSolver',counted)
    problem=dict(A_ub=np.zeros((0,3)),b_ub=np.empty(0),A_eq=np.array([[1.,1.,0.]]),
                 b_eq=np.array([1.]),bounds=[(0.,1.),(0.,1.),(.2,.3)])
    cache={}
    for feasible in [np.array([.8,.2,.25]),np.array([.7,.3,.25])]:
        x,note=module.minimum_input(problem,feasible,np.array([0,1]),np.array([20.,20.]),.2,method='conic',_direct_cache=cache)
        assert note['accepted'] and note['normalized_residual']<1e-8
        np.testing.assert_allclose(x[:2],[.5,.5],atol=1e-5)
    assert len(calls)==1
    problem['b_eq'][0]=1.1
    x,note=module.minimum_input(problem,np.array([.7,.4,.25]),np.array([0,1]),np.array([20.,20.]),.2,method='conic',_direct_cache=cache)
    assert len(calls)==2 and note['accepted']
    np.testing.assert_allclose(x[:2],[.55,.55],atol=1e-5)
