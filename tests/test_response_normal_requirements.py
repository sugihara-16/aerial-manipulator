import numpy as np
import pytest
from build.response_contact_control import normal_load_violation


def test_different_pad_requirements_preserve_each_physical_lower_bound():
    assert normal_load_violation([.5, 1.], [.5, 2.]) == .5
    assert normal_load_violation([2., .25], [.5, .5]) == .5
    assert normal_load_violation([.5, 2.], [.5, 2.]) == 0.


def test_scalar_and_uniform_vector_define_same_condition():
    for loads in [[0., 0.], [.3, .8], [1., 2.]]:
        assert normal_load_violation(loads, .5) == normal_load_violation(loads, [.5, .5])
    assert normal_load_violation([0., .5], [0., .5]) == 0.
    assert normal_load_violation([0., 0.], 0.) == 0.


def test_invalid_requirements_are_not_silently_ignored():
    for floor in [[-.1, .5], [np.nan, .5], [np.inf, .5]]:
        with pytest.raises(ValueError):normal_load_violation([1.,1.], floor)
