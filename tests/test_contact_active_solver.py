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
