"""Nonoverlapping process windows and transit replacement contracts."""

import pytest
from assemble_horizons import HorizonAssemblyError, assemble_horizons, eligible_windows


def test_windows_reuse_no_process_stage_and_keep_ineligible_transits():
    labels = [
        "withdraw:0",
        "transit:1",
        "approach:1",
        "search:1",
        "withdraw:1",
        "transit:2",
        "approach:2",
        "transit:3",
    ]
    assert eligible_windows(labels) == [
        ("withdraw:0", "transit:1", "approach:1"),
        ("withdraw:1", "transit:2", "approach:2"),
    ]
    baseline = [
        {
            "stage": f"transit:{i}",
            "status": "sampled_validated",
            "curve": {"stage": f"transit:{i}", "old": True},
        }
        for i in (1, 2, 3)
    ]
    selected = [
        {
            "stage_sequence": w,
            "selected": {"rates": [0.5, 1.5], "transit": {"stage": w[1], "new": True}},
        }
        for w in eligible_windows(labels)
    ]
    summary, rates = assemble_horizons(labels, baseline, selected)
    assert [x["stage"] for x in summary] == ["transit:1", "transit:2", "transit:3"]
    assert summary[0]["curve"]["new"] is True
    assert summary[-1]["curve"]["old"] is True
    assert rates == {
        "withdraw:0": (0.0, 0.5),
        "approach:1": (1.5, 0.0),
        "withdraw:1": (0.0, 0.5),
        "approach:2": (1.5, 0.0),
    }
    assert baseline[0]["curve"]["old"] is True


def test_missing_duplicate_and_mismatched_window_fail_loudly():
    labels = ["withdraw:0", "transit:1", "approach:1"]
    baseline = [
        {"stage": "transit:1", "status": "sampled_validated", "curve": {"stage": "transit:1"}}
    ]
    report = {
        "stage_sequence": labels,
        "selected": {"rates": [1, 1], "transit": {"stage": "transit:1"}},
    }
    with pytest.raises(HorizonAssemblyError):
        assemble_horizons(labels, baseline, [])
    with pytest.raises(HorizonAssemblyError):
        assemble_horizons(labels, baseline, [report, report])
    report["selected"]["transit"]["stage"] = "transit:2"
    with pytest.raises(HorizonAssemblyError):
        assemble_horizons(labels, baseline, [report])
