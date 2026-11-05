import numpy as np
import pytest
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve


class Checker:
    def __init__(self, colliding=False):
        self.colliding = colliding

    def contacts(self, _state):
        return [type("Contact", (), {"distance_m": -0.01})()] if self.colliding else []


def test_inspects_all_nine_axes_and_preserves_raw_excursion():
    limits = np.tile([-1.0, 1.0], (9, 1))
    values = np.zeros((3, 9))
    values[1, 0] = 1.00005
    result = inspect_sampled_curve(Checker(), values, limits, 0.0001)
    assert result["sample_count"] == 3
    assert result["max_raw_limit_excursion_by_axis"][0] == pytest.approx(0.00005)
    values[1, 8] = 1.1
    with pytest.raises(CurveLimitError):
        inspect_sampled_curve(Checker(), values, limits, 0.0001)


def test_rejects_actual_sample_contact():
    with pytest.raises(CurveCollisionError):
        inspect_sampled_curve(Checker(True), np.zeros((2, 9)), np.tile([-1.0, 1.0], (9, 1)), 0.0001)
