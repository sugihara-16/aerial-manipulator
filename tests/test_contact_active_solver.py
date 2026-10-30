import json
from pathlib import Path
import clarabel
import numpy as np
import pytest
from build.contact_active_solver import solve


def test_norm_boundary_requires_cuts_and_retains_known_optimum():
    # Project (3, 4) onto the unit disk. A box-only relaxation is insufficient.
    matrix = np.vstack([np.eye(2), np.zeros((1, 2)), -np.eye(2)])
    bound = np.array([10., 10., 1., 0., 0.])
    cones = [clarabel.ZeroConeT(0), clarabel.NonnegativeConeT(2), clarabel.SecondOrderConeT(3)]
    x, info = solve(np.eye(2), np.array([-3., -4.]), matrix, bound, cones)
    np.testing.assert_allclose(x, [.6, .8], atol=3e-7)
    assert np.linalg.norm(x) <= 1 + 1e-10
    assert info['cut_rounds'] > 0 and info['objective_gap'] < 1e-7


def test_saved_ill_scaled_contact_problem_keeps_original_constraints():
    path = Path(__file__).parent/'fixtures/contact_equality_scaling.npz'
    data = np.load(path)
    cones = [getattr(clarabel, x['type'])(x['dim']) for x in json.loads(path.with_suffix('.json').read_text())]
    x, info = solve(*(data[k] for k in ['P', 'c', 'A', 'b']), cones)
    assert info['original_residual'] <= 1e-7
    assert abs(.5*x@data['P']@x + data['c']@x - .464758323) < 1e-6


def test_inconsistent_equations_are_rejected():
    with pytest.raises(ValueError, match='inconsistent'):
        solve(np.eye(2), np.zeros(2), np.array([[1., 0.], [1., 0.]]),
              np.array([1., 2.]), [clarabel.ZeroConeT(2)])


def test_warm_active_constraints_are_rechecked_when_optimum_moves():
    from types import SimpleNamespace
    workspace=SimpleNamespace()
    A=np.r_[np.eye(3),-np.eye(3)];b=np.ones(6)
    cones=[clarabel.ZeroConeT(0),clarabel.NonnegativeConeT(6)]
    for c in [np.array([-3.,-.4,2.]),np.array([-3.1,-.3,2.1]),np.array([.2,3.,-.1])]:
        actual,note=solve(np.eye(3),c,A,b,cones,workspace=workspace)
        expected,_=solve(np.eye(3),c,A,b,cones)
        np.testing.assert_allclose(actual,expected,atol=1e-7)
        assert note['original_residual']<1e-7 and note['objective_gap']<1e-7


def test_velocity_elimination_retains_redundant_reaction_modes():
    from types import SimpleNamespace
    rng=np.random.default_rng(207);nv,nu,nr=3,2,2;n=nv+nu+nr
    mass=np.diag([1.,2.,3.]);B=rng.normal(size=(nv,nu));C=np.array([[1.,1.,0.],[1.,1.,0.]])
    E=np.block([[mass,-B,-C.T],[C,np.zeros((nr,nu+nr))]])
    A=np.r_[E,np.eye(n),-np.eye(n)];b=np.r_[np.zeros(len(E)),np.full(2*n,2.)]
    cones=[clarabel.ZeroConeT(len(E)),clarabel.NonnegativeConeT(2*n)]
    workspace=SimpleNamespace()
    for _ in range(15):
        R=rng.normal(size=(n,n));P=R.T@R+.1*np.eye(n);c=rng.normal(size=n)
        actual,note=solve(P,c,A,b,cones,mechanical_partition=(nv,nu),workspace=workspace)
        expected,_=solve(P,c,A,b,cones)
        np.testing.assert_allclose(actual,expected,atol=1e-7)
        assert note['original_residual']<1e-7


def test_shared_mechanics_are_reused_only_for_identical_equations():
    rng=np.random.default_rng(9);nv,nu=3,4;n=nv+nu
    E=np.c_[np.diag([1.,2.,3.]),rng.normal(size=(nv,nu))]
    A=np.r_[E,np.eye(n),-np.eye(n)];b=np.r_[np.zeros(nv),np.ones(2*n)]
    cones=[clarabel.ZeroConeT(nv),clarabel.NonnegativeConeT(2*n)];cache={'base':n};P=np.eye(n);c=rng.normal(size=n)
    for change in [0.,0.,.01,0.]:
        matrix=A.copy();matrix[0,0]+=change
        x,_=solve(P,c,matrix,b,cones,mechanical_partition=(nv,nu),mechanical_cache=cache)
        reference,_=solve(P,c,matrix,b,cones,mechanical_partition=(nv,nu))
        np.testing.assert_allclose(x,reference,atol=1e-10)
