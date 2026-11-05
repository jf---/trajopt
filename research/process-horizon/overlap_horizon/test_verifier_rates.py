"""Whole-tour verifier preserves the default process clock and validates overrides."""

import pytest
from verify_duration_policy import ProcessRateError, process_rates_for_stage


def test_default_and_explicit_shared_process_rates():
    assert process_rates_for_stage("withdraw:0", {}) == (0.0, 1.0)
    assert process_rates_for_stage("approach:1", {}) == (1.0, 0.0)
    assert process_rates_for_stage("search:1", {}) == (0.0, 0.0)
    assert process_rates_for_stage("withdraw:0", {"withdraw:0": (0.0, 0.5)}) == (0.0, 0.5)
    assert process_rates_for_stage("approach:1", {"approach:1": (0.5, 0.0)}) == (0.5, 0.0)


def test_process_corner_override_and_invalid_rates_fail_loudly():
    with pytest.raises(ProcessRateError):
        process_rates_for_stage("search:1", {"search:1": (1.0, 0.0)})
    with pytest.raises(ProcessRateError):
        process_rates_for_stage("withdraw:0", {"withdraw:0": (0.5, 1.0)})
    with pytest.raises(ProcessRateError):
        process_rates_for_stage("approach:1", {"approach:1": (-1.0, 0.0)})
