"""Numerical restarts must preserve the problem and never bypass acceptance."""
from types import SimpleNamespace
import numpy as np
import pytest
from build.response_contact_control import ResponseContactControl

UNRESOLVED = 'complete contact inverse found no feasible held input: test'


def controller():
    control = ResponseContactControl.__new__(ResponseContactControl)
    control.predictor = SimpleNamespace(initial_step=np.array([.0001, .002]),
        minimum_step=np.array([.000006, .00012]), epsilon=np.array([.00001, .0002]),
        cached=object(), origin=object())
    control.previous = np.array([.3, .7]); control.damping = .001
    control.diagnostics = {}
    return control


@pytest.mark.parametrize('accepted_factor', [1, 10, 100, None])
def test_only_verified_result_is_returned_and_resolution_is_not_carried_over(accepted_factor):
    control = controller(); predictor = control.predictor
    initial = predictor.initial_step.copy(); minimum = predictor.minimum_step.copy()
    q, v, target, velocity, anchor, normal = [np.array([i]) for i in range(6)]
    calls = []
    def solve(*args):
        assert all(x is y for x, y in zip(args, (q, v, target, velocity, anchor, normal)))
        np.testing.assert_array_equal(control.previous, [.3, .7])
        assert control.damping == .001
        factor = round(float(predictor.initial_step[0]/initial[0])); calls.append(factor)
        if factor == accepted_factor:
            control.previous = np.array([.4, .8])
            return control.previous.copy()
        control.previous = np.array([99., 99.]); control.damping = 1e8
        raise ValueError(UNRESOLVED)
    control._update = solve
    if accepted_factor is None:
        with pytest.raises(ValueError, match='no feasible held input'):
            control.update(q, v, target, velocity, anchor, normal)
        assert calls == [1, 10, 100]
    else:
        np.testing.assert_array_equal(control.update(q, v, target, velocity, anchor, normal), [.4, .8])
        assert calls == [1, 10, 100][:calls.index(accepted_factor)+1]
    np.testing.assert_array_equal(predictor.initial_step, initial)
    np.testing.assert_array_equal(predictor.minimum_step, minimum)
    if len(calls) > 1:
        assert predictor.cached is None and predictor.origin is None
        np.testing.assert_array_equal(predictor.epsilon, initial)
    if accepted_factor in (10, 100):
        assert control.secant_retry_recovered == 1
        assert control.diagnostics['secant_retry_factor'] == accepted_factor


def test_non_numerical_error_is_not_retried():
    control = controller(); calls = []
    def fail(*args):
        calls.append(1)
        raise ValueError('observed state has an empty initial input domain')
    control._update = fail
    with pytest.raises(ValueError, match='empty initial input domain'):
        control.update(None, None, None)
    assert len(calls) == 1
