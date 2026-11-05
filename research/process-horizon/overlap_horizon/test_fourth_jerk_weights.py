"""A global dimensionless risk weight preserves the old zero-weight path."""

import numpy as np
import pytest
from native_accel_transits import TransitContractError, fourth_jerk_weights


def test_zero_weight_removes_new_cost_and_positive_weight_has_derived_units():
    speeds = np.array([2.0, 4.0])
    assert fourth_jerk_weights(speeds, 0.0, None) is None
    result = fourth_jerk_weights(speeds, 0.25, 2.0)
    np.testing.assert_allclose(result, 1e-6 * 0.25 * 2.0**4 / speeds**4)


def test_invalid_risk_profile_fails_loudly():
    for weight, duration in ((-1.0, 1.0), (1.0, None), (1.0, 0.0), (float("nan"), 1.0)):
        with pytest.raises(TransitContractError):
            fourth_jerk_weights(np.array([2.0]), weight, duration)
