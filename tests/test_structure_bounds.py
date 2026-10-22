"""Check universal count bounds independently of the structure search."""
import numpy as np
from amsrr_det.structure_bounds import count_lower_bounds, compare_count_groups, hover_input_burden


def test_weight_bound_holds_for_arbitrary_force_directions_and_unequal_thrust():
    rng = np.random.default_rng(22645)
    for n in range(1, 9):
        for _ in range(10):
            force = rng.uniform(0, 20, 4*n)
            directions = rng.normal(size=(4*n, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            weight = np.linalg.norm(force@directions)
            assert weight**1.5/np.sqrt(4*n) <= hover_input_burden(force)


def test_group_comparison_does_not_exclude_unresolved_low_counts():
    bounds = count_lower_bounds(1.8667403011, .4438785179843946, 9.80665)
    rows = compare_count_groups(bounds, 113.56199657)
    assert [r['modules'] for r in rows if r['status']=='excluded_by_weight'] == list(range(3, 9))
    assert all(r['maximum_improvement'] > 0 for r in rows[:2])


def test_large_payload_can_make_more_modules_reduce_the_lower_bound():
    bounds = count_lower_bounds(2., 20., 9.80665)
    assert np.argmin([r['lower_bound'] for r in bounds])+1 == 5
    assert bounds[1]['lower_bound'] < bounds[0]['lower_bound']


def test_finite_balance_residual_relaxes_the_weight_lower_bound():
    exact=count_lower_bounds(2.,.8,9.8)
    tolerance=1e-3
    relaxed=count_lower_bounds(2.,.8,9.8,force_balance_tolerance=tolerance)
    for n,(old,new) in enumerate(zip(exact,relaxed),start=1):
        assert new['lower_bound']<old['lower_bound']
        # Both bodies' residual vectors can align with gravity with this norm.
        thrust=((2*n+.8)*9.8-2*np.sqrt(3)*tolerance)/(4*n)
        np.testing.assert_allclose(new['lower_bound'],hover_input_burden(np.full(4*n,thrust)),atol=1e-12)
