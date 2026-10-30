"""Deadline output keeps physical bounds, including after native interruption."""
import time
import numpy as np
import quadprog
from amsrr_det.bounded_qp import bounded_qp


def problem(nu=4, pads=2):
    n = nu+3*pads
    lower, upper = -np.ones(nu), np.ones(nu)
    C = np.zeros((n, 2*nu+9*pads))
    C[:nu, :2*nu] = np.c_[np.eye(nu), -np.eye(nu)]
    angles = np.arange(8)*np.pi/4
    for i in range(pads):
        row, col = nu+3*i, 2*nu+9*i
        C[row, col] = 1.
        C[row, col+1:col+9] = 2.25*np.cos(np.pi/8)
        C[row+1:row+3, col+1:col+9] = -np.array([np.cos(angles), np.sin(angles)])
    return C, np.r_[lower, -upper, np.zeros(9*pads)], lower, upper


def test_normal_solution_matches_original_solver():
    C, b, lower, upper = problem()
    rng = np.random.default_rng(1028)
    A = rng.normal(size=(30, len(C)))
    H, a = A.T@A+.1*np.eye(len(C)), rng.normal(size=len(C))*10
    x, expired, _ = bounded_qp(H, a, C, b, lower, upper, np.zeros(len(C)),
                              deadline=time.perf_counter()+1.)
    assert not expired
    np.testing.assert_allclose(x, quadprog.solve_qp(H, a, C, b)[0], atol=1e-10)


def test_expired_output_projects_motors_and_friction_and_does_not_leak_deadline():
    C, b, lower, upper = problem()
    previous = np.array([10., -20., np.nan, np.inf, -3., 2., 2., 1., 10., -10.])
    H, a = np.eye(len(C)), np.zeros(len(C))
    x, expired, _ = bounded_qp(H, a, C, b, lower, upper, previous,
                              deadline=time.perf_counter()-1.)
    assert expired and np.isfinite(x).all()
    assert min(C.T@x-b) >= -1e-12
    _, expired, _ = bounded_qp(H, a, C, b, lower, upper, x,
                              deadline=time.perf_counter()+1.)
    assert not expired


def test_native_deadline_interrupts_a_large_problem_without_invalid_output():
    C, b, lower, upper = problem(64, 30)
    rng = np.random.default_rng(74)
    A = rng.normal(size=(200, len(C)))
    H, a = A.T@A+.01*np.eye(len(C)), rng.normal(size=len(C))*100
    x, expired, _ = bounded_qp(H, a, C, b, lower, upper, np.zeros(len(C)),
                              deadline=time.perf_counter()+.0001)
    assert expired
    assert np.isfinite(x).all() and min(C.T@x-b) >= -1e-8
