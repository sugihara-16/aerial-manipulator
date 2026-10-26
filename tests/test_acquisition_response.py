from unittest.mock import Mock
import pytest
from build.contact_step_control import ContactStepControl


def controller(message, supported=False):
    control=ContactStepControl.__new__(ContactStepControl)
    control._update=Mock(side_effect=ValueError(message))
    control.object_velocity_bounds=(0,1) if supported else None
    control.object_velocity_norm_limits=None
    control.separation=object()
    control._held_response=Mock(return_value='validated command')
    return control


def test_finite_response_is_only_used_for_known_affine_contact_failures():
    control=controller('contact friction branches unresolved: test')
    assert control.update(1,2,3,4,5,6)=='validated command'
    control._held_response.assert_called_once_with(1,2,3,4,5,6)


def test_support_speed_constraints_are_never_silently_dropped():
    control=controller('constant constraint infeasible',supported=True)
    with pytest.raises(ValueError):control.update(1,2,3)
    control._held_response.assert_not_called()


def test_support_speed_norm_is_never_dropped_by_response_fallback():
    control=controller('constant constraint infeasible')
    control.object_velocity_norm_limits=(.02,.1)
    with pytest.raises(ValueError):control.update(1,2,3)
    control._held_response.assert_not_called()


def test_other_failures_remain_failures():
    control=controller('invalid geometry')
    with pytest.raises(ValueError,match='invalid geometry'):control.update(1,2,3)
    control._held_response.assert_not_called()
