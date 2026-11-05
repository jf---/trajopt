"""Assemble nonoverlapping probe-transit windows onto an existing baseline."""

import copy

import numpy as np


class HorizonAssemblyError(RuntimeError):
    """The windows or baseline cannot form one unambiguous tour."""


def eligible_windows(labels):
    """Find every withdrawal-transit-approach triple, with no shared stage."""
    if len(labels) != len(set(labels)):
        raise HorizonAssemblyError("stage labels must be unique")
    windows = [
        (labels[i - 1], labels[i], labels[i + 1])
        for i in range(1, len(labels) - 1)
        if labels[i - 1].startswith("withdraw:")
        and labels[i].startswith("transit:")
        and labels[i + 1].startswith("approach:")
    ]
    members = [label for window in windows for label in window]
    if len(members) != len(set(members)):
        raise HorizonAssemblyError("candidate windows overlap")
    return windows


def assemble_horizons(labels, baseline, reports):
    """Replace selected transits and build probe-rate overrides without mutation."""
    windows = eligible_windows(labels)
    actual = [tuple(x["stage_sequence"]) for x in reports]
    if (
        len(actual) != len(windows)
        or len(set(actual)) != len(actual)
        or set(actual) != set(windows)
    ):
        raise HorizonAssemblyError("every eligible window needs exactly one result")
    expected_transits = [label for label in labels if label.startswith("transit:")]
    if [x["stage"] for x in baseline] != expected_transits:
        raise HorizonAssemblyError("baseline transit sequence mismatch")
    by_stage = {entry["stage"]: copy.deepcopy(entry) for entry in baseline}
    rates = {}
    for report in reports:
        left, transit, right = report["stage_sequence"]
        selected = report["selected"]
        pair = np.asarray(selected["rates"], dtype=float)
        if pair.shape != (2,) or not np.isfinite(pair).all() or np.any(pair <= 0):
            raise HorizonAssemblyError("selected rates must be finite and positive")
        if selected["transit"]["stage"] != transit:
            raise HorizonAssemblyError("selected transit label mismatch")
        by_stage[transit] = {
            "stage": transit,
            "status": "sampled_validated",
            "curve": copy.deepcopy(selected["transit"]),
        }
        rates[left] = (0.0, float(pair[0]))
        rates[right] = (float(pair[1]), 0.0)
    return [by_stage[stage] for stage in expected_transits], rates
