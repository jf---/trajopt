"""Splice generated process macros and derive neighboring transit endpoints."""

import copy

import numpy as np


class RiskSegmentsError(RuntimeError):
    """The generated macros do not match the source stage layout."""


def build_risk_segments(segments, candidate):
    """Return a copied stage sequence using generated macros, never recorded joints."""
    result = copy.deepcopy(segments)
    labels = [item["stage"] for item in result]
    if len(labels) != len(set(labels)):
        raise RiskSegmentsError("stage labels must be unique")
    macros = candidate["macros"]
    seen = set()
    for macro in macros:
        number = int(macro["id"].split(":")[0])
        if number in seen:
            raise RiskSegmentsError("duplicate generated probe macro")
        seen.add(number)
        states = np.asarray(macro["path"], dtype=float)
        if states.shape != (65, 7) or not np.isfinite(states).all():
            raise RiskSegmentsError("generated probe macro must have 65 finite seven-axis states")
        for index, phase in enumerate(("approach", "search", "retract", "withdraw")):
            name = f"{phase}:{number}"
            if name not in labels:
                raise RiskSegmentsError(f"generated process stage is absent: {name}")
            item = result[labels.index(name)]
            source = np.asarray(item["states"], dtype=float)
            fixture = source[0, 7:]
            if source.ndim != 2 or source.shape[1] != 9 or not np.all(source[:, 7:] == fixture):
                raise RiskSegmentsError(f"fixture changes inside {name}")
            part = states[index * 16 : index * 16 + 17]
            item["states"] = np.column_stack((part, np.tile(fixture, (len(part), 1)))).tolist()
    changed = [item["solution"]["stage"] for item in candidate["transits"]]
    if len(changed) != len(set(changed)):
        raise RiskSegmentsError("duplicate candidate transit")
    for name in changed:
        if name not in labels or not name.startswith("transit:"):
            raise RiskSegmentsError(f"unknown transit {name}")
        i = labels.index(name)
        if i == 0 or i == len(result) - 1:
            raise RiskSegmentsError("affected transit must have both neighboring stages")
        left = result[i - 1]["states"][-1]
        right = result[i + 1]["states"][0]
        if left[7:] != right[7:]:
            raise RiskSegmentsError(f"fixture changes inside {name}")
        result[i]["states"] = [left.copy(), right.copy()]
    return result
