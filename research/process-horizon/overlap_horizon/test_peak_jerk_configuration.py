"""Peak risk normalization and zero-weight compatibility."""

import numpy as np
import pytest
from native_accel_transits import TransitContractError, peak_jerk_configuration


def test_zero_weight_preserves_original_problem_and_positive_weight_has_units():
    speeds = np.array([2.0, 4.0])
    assert peak_jerk_configuration(speeds, 0.0, None) is None
    normalization, coefficients = peak_jerk_configuration(speeds, 0.25, 2.0)
    np.testing.assert_allclose(normalization, 2.0**2 / speeds)
    np.testing.assert_allclose(coefficients, np.full(2, 1e-6 * 0.25 / 2.0**3))


def test_peak_profile_rejects_missing_or_invalid_scale():
    for weight, reference in ((-1.0, 1.0), (1.0, None), (1.0, 0.0), (float("nan"), 1.0)):
        with pytest.raises(TransitContractError):
            peak_jerk_configuration(np.array([2.0]), weight, reference)
