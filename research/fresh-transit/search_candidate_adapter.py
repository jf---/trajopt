"""Validate optimized SearchL candidates against generated boundaries."""

from typing import Any

import numpy as np

# Numeric reconstruction parity, not a path or manufacturing tolerance.
SEARCH_PARITY_ULPS = 32768


class SearchCandidateError(RuntimeError):
    """Generated and optimized SearchL boundary contracts differ."""


def _same(a: Any, b: Any) -> bool:
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    band = (
        SEARCH_PARITY_ULPS
        * np.finfo(float).eps
        * max(
            1.0, float(np.max(np.abs(left), initial=0.0)), float(np.max(np.abs(right), initial=0.0))
        )
    )
    return bool(np.all(np.abs(left - right) <= band))


def checked_search_nodes(
    process: dict[str, Any], probe: dict[str, Any], evidence: dict[str, Any], index: int
) -> Any:
    """Return generated candidate nodes only after geometry and boundary provenance checks."""
    if (
        process["stage"] != f"search:{index}"
        or probe["probe"] != index
        or not probe["scene_restored"]
        or len(probe["variants"]) != 1
    ):
        raise SearchCandidateError("search labels, ordering or scene state differ")
    variant = probe["variants"][0]
    if (
        variant["line_mode"] != "hard"
        or variant["progress_weight"] != 1.0
        or variant["status"] != "SQPStatus.NLP_CONVERGED"
        or variant["actual_curve"]["penetrating_contacts"]
    ):
        raise SearchCandidateError("search candidate lacks the required sampled checks")
    original = np.asarray(process["ik_states"], dtype=float)
    candidate = np.asarray(variant["candidate_nodes"], dtype=float)
    if (
        original.ndim != 2
        or original.shape[1] != 7
        or candidate.shape != original.shape
        or not np.isfinite(candidate).all()
        or not _same(original, evidence["generated_input_nodes"])
        or not _same(candidate[[0, -1]], original[[0, -1]])
    ):
        raise SearchCandidateError("generated states or optimized endpoint states differ")
    if not _same([process["duration_s"]], [evidence["modeled_duration_s"]]):
        raise SearchCandidateError("modeled search clocks differ")
    if any(
        not _same(process[key], evidence["boundary_velocity_acceleration"][key])
        for key in ("v0", "a0", "v1", "a1")
    ):
        raise SearchCandidateError("modeled endpoint derivatives differ")
    return candidate
