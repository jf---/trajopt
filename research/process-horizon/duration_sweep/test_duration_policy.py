"""Contract for one shared transit clock and preserved process/fixture semantics."""

import copy

import numpy as np
import pytest
from duration_policy import DurationPolicyError, scale_transit_clock, validate_transit_results


def fixture():
    stages = [
        {"stage": "transit:0", "states": [[0.0] * 9, [0.0] * 9]},
        {"stage": "weld:1", "states": [[0.0] * 9, [0.0] * 9]},
        {"stage": "fixture:2", "states": [[0.0] * 9, [0.0] * 9]},
        {"stage": "fixture:3", "states": [[0.0] * 9, [0.0] * 9]},
        {"stage": "transit:4", "states": [[0.0] * 9, [0.0] * 9]},
    ]
    clock = {
        "stages": [
            {"stage": s["stage"], "duration_s": d}
            for s, d in zip(stages, [1.0, 3.0, 0.2, 0.3, 2.0], strict=True)
        ]
    }
    return stages, clock


def test_uniform_transits_preserve_process_and_fixture_total_without_mutation():
    stages, clock = fixture()
    original = copy.deepcopy(clock)
    result = scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.5])
    assert [x["duration_s"] for x in result["stages"]] == [2.0, 3.0, 0.1, 0.4, 4.0]
    assert clock == original


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf"), True])
def test_reject_invalid_uniform_factor(factor):
    stages, clock = fixture()
    with pytest.raises(DurationPolicyError):
        scale_transit_clock(stages, clock, factor, [0.0, 0.1, 0.5])


def test_reject_fixture_cycle_change_and_stage_label_mismatch():
    stages, clock = fixture()
    with pytest.raises(DurationPolicyError):
        scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.6])
    clock["stages"][1]["stage"] = "different"
    with pytest.raises(DurationPolicyError):
        scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.5])


def test_transit_results_require_complete_success_matching_time_and_fixture():
    stages, clock = fixture()
    policy = scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.5])
    results = [
        {
            "stage": label,
            "status": "sampled_validated",
            "curve": {
                "stage": label,
                "duration_s": duration,
                "knots_s": [0.0, duration],
                "fixture": [0.0, 0.0],
            },
        }
        for label, duration in [("transit:0", 2.0), ("transit:4", 4.0)]
    ]
    assert set(validate_transit_results(stages, policy, results)) == {"transit:0", "transit:4"}
    for bad in (
        results[:-1],
        [results[0], results[0]],
        [{**results[0], "status": "failed"}, results[1]],
    ):
        with pytest.raises(DurationPolicyError):
            validate_transit_results(stages, policy, bad)
    changed = copy.deepcopy(results)
    changed[0]["curve"]["fixture"] = [np.float64(1.0), 0.0]
    with pytest.raises(DurationPolicyError):
        validate_transit_results(stages, policy, changed)


def test_malformed_stage_and_curve_fail_with_named_error():
    stages, clock = fixture()
    stages[0]["stage"] = None
    with pytest.raises(DurationPolicyError):
        scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.5])
    stages, clock = fixture()
    policy = scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.5])
    with pytest.raises(DurationPolicyError):
        validate_transit_results(stages, policy, [{"status": "sampled_validated"}] * 2)


def test_fixture_retiming_preserves_every_nine_axis_event():
    from duration_policy import validate_fixture_policy

    stages, _ = fixture()
    start = np.zeros(9)
    end = np.zeros(9)
    end[7] = 1.0
    progress = 10 * 0.2**3 - 15 * 0.2**4 + 6 * 0.2**5
    mid = start + progress * (end - start)
    stages[2]["states"] = [start.tolist(), mid.tolist()]
    stages[3]["states"] = [mid.tolist(), end.tolist()]
    policy = {
        "start": start.tolist(),
        "end": end.tolist(),
        "stage_times_s": [0.0, 0.1, 0.5],
        "duration_s": 0.5,
    }
    assert validate_fixture_policy(stages, policy) == pytest.approx(0.0, abs=1e-15)
    policy["end"][7] = 2.0
    with pytest.raises(DurationPolicyError):
        validate_fixture_policy(stages, policy)


def test_zero_duration_event_fails_instead_of_inventing_a_clock():
    stages, clock = fixture()
    clock["stages"][1]["duration_s"] = 0.0
    with pytest.raises(DurationPolicyError, match="positive"):
        scale_transit_clock(stages, clock, 2.0, [0.0, 0.1, 0.5])
