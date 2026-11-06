"""One source-geometry fixture chain can be eased without changing its events."""

import numpy as np
import pytest
from actual_stage_contract import diagnostic_joins
from fixture_easing_policy import FixtureEasingError, ease_fixture_chains


def stages():
    q0 = np.zeros(9)
    q1 = q0.copy()
    q1[7] = 0.3
    q2 = q0.copy()
    q2[7] = 1.0
    segments = [
        {"stage": "transit:0", "states": [q0.tolist(), q0.tolist()]},
        {"stage": "fixture:1", "states": [q0.tolist(), q1.tolist()]},
        {"stage": "fixture:2", "states": [q1.tolist(), q2.tolist()]},
        {"stage": "transit:3", "states": [q2.tolist(), q2.tolist()]},
    ]
    clock = {
        "stages": [
            {"stage": x["stage"], "duration_s": v}
            for x, v in zip(segments, [1.0, 1.0, 2.0, 1.0], strict=True)
        ]
    }
    return segments, clock


def test_eases_all_ordered_targets_without_intermediate_stop():
    source, clock = stages()
    timing, curves = ease_fixture_chains(source, clock)
    assert len(curves) == 2
    assert sum(x["duration_s"] for x in timing["stages"][1:3]) == pytest.approx(3.0)
    assert curves["fixture:1"].evaluate(curves["fixture:1"].duration).position[7] == pytest.approx(
        0.3
    )
    assert curves["fixture:1"].evaluate(curves["fixture:1"].duration).velocity[7] > 0
    assert curves["fixture:1"].evaluate(0.0).velocity == pytest.approx((0.0,) * 9)
    assert curves["fixture:2"].evaluate(curves["fixture:2"].duration).velocity == pytest.approx(
        (0.0,) * 9
    )
    join = diagnostic_joins([curves["fixture:1"], curves["fixture:2"]])[0]
    assert join.velocity_delta == pytest.approx((0.0,) * 9)
    assert join.acceleration_delta == pytest.approx((0.0,) * 9)
    assert clock["stages"][1]["duration_s"] == 1.0


def test_noncollinear_fixture_path_is_explicitly_unsupported():
    source, clock = stages()
    source[2]["states"][-1][0] = 0.1
    with pytest.raises(FixtureEasingError, match="fixture"):
        ease_fixture_chains(source, clock)


def test_adjacent_transit_uses_actual_eased_fixture_endpoint_velocity():
    from fixture_easing_policy import eased_boundary_resolver

    source, clock = stages()
    timing, curves = ease_fixture_chains(source, clock)
    resolver = eased_boundary_resolver(curves)
    left, right, _ = resolver(
        source, timing["stages"], 0, start_velocity=np.zeros(9), end_velocity=np.zeros(9)
    )
    assert left == pytest.approx((0.0,) * 9)
    assert right == pytest.approx((0.0,) * 9)
    left, right, _ = resolver(
        source, timing["stages"], 3, start_velocity=np.zeros(9), end_velocity=np.zeros(9)
    )
    assert left == pytest.approx((0.0,) * 9)
    assert right == pytest.approx((0.0,) * 9)


def test_audit_uses_fixture_curves_instead_of_linear_source_derivatives():
    from fresh_actual_audit import audit_stages

    source, clock = stages()
    timing, curves = ease_fixture_chains(source, clock)
    fixture = source[1:3]
    selected_clock = {"stages": timing["stages"][1:3]}
    report = audit_stages(fixture, selected_clock, [], lambda _: None, stage_overrides=curves)
    internal = report["joins"][1]
    assert internal.left == "fixture:1" and internal.right == "fixture:2"
    assert internal.velocity_delta == pytest.approx((0.0,) * 9)
    assert internal.acceleration_delta == pytest.approx((0.0,) * 9)
