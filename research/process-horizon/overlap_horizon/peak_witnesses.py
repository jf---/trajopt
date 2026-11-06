"""Select actual-spline peak epigraph witnesses with explicit SQP tolerance."""

import numpy as np
from spline_speed_extrema import jerk_extrema

# Roundoff band for deduplicating modeled seconds, not a physical tolerance.
TIME_ROUNDOFF_ULPS = 128


class PeakWitnessError(RuntimeError):
    """The peak epigraph input is invalid or a violating witness cannot be added."""


def select_peak_witnesses(curve, peaks, normalization, existing_times_s, feasibility_band):
    """Return times of numerical jerk extrema outside the optimized epigraph.

    `peaks` and `feasibility_band` are dimensionless; `normalization` is the
    positive reference-time-squared divided by sourced axis speed. The band is
    the explicitly configured SQP constraint feasibility setting, not a
    physical jerk bound. All numerical extrema are returned as evidence.
    """
    values = np.asarray(peaks, dtype=float)
    scale = np.asarray(normalization, dtype=float)
    existing = np.asarray(existing_times_s, dtype=float)
    extrema = jerk_extrema(curve)
    if (
        values.shape != (len(extrema),)
        or scale.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or np.any(values < -feasibility_band)
        or existing.ndim != 1
        or not np.isfinite(existing).all()
        or not np.isfinite(feasibility_band)
        or feasibility_band < 0
    ):
        raise PeakWitnessError(
            "finite dimensionless peak, scale, witnesses and feasibility band required"
        )
    added = []
    evidence = []
    for axis, item in enumerate(extrema):
        excess = float(item["absolute_jerk"] * scale[axis] - values[axis])
        time = float(item["time_s"])
        evidence.append({"axis": axis, **item, "normalized_excess": excess})
        if excess <= feasibility_band:
            continue
        band = TIME_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, abs(time))
        if np.any(np.abs(existing - time) <= band):
            raise PeakWitnessError(f"violating extremum already constrained on axis {axis}")
        if not any(abs(time - earlier) <= band for earlier in added):
            added.append(time)
    return sorted(added), evidence


# One shared finite refinement budget for every transit.
DEFAULT_MAX_WITNESS_ROUNDS = 5


def _refine_peak_witnesses(initial_times_s, solve, normalization, feasibility_band, max_rounds):
    witnesses = sorted(map(float, initial_times_s))
    if (
        not witnesses
        or len(set(witnesses)) != len(witnesses)
        or not isinstance(max_rounds, int)
        or isinstance(max_rounds, bool)
        or max_rounds <= 0
    ):
        raise PeakWitnessError("unique initial witnesses and positive finite budget required")
    previous = None
    rounds = []
    for _ in range(max_rounds):
        result = solve(witnesses, previous)
        added, evidence = select_peak_witnesses(
            result["curve"], result["peak_epigraph"], normalization, witnesses, feasibility_band
        )
        rounds.append({"witnesses_s": witnesses.copy(), "added_s": added, "extrema": evidence})
        if not added:
            return result, rounds, "satisfied"
        witnesses = sorted([*witnesses, *added])
        previous = result
    return result, rounds, "budget_exhausted"


def refine_peak_witnesses(
    initial_times_s,
    solve,
    normalization,
    feasibility_band,
    *,
    max_rounds=DEFAULT_MAX_WITNESS_ROUNDS,
):
    """Require the bounded witness refinement to satisfy its configured band."""
    result, rounds, status = _refine_peak_witnesses(
        initial_times_s, solve, normalization, feasibility_band, max_rounds
    )
    if status != "satisfied":
        raise PeakWitnessError(f"peak witness budget exhausted: {rounds}")
    return result, rounds


def refine_peak_witnesses_bounded(
    initial_times_s,
    solve,
    normalization,
    feasibility_band,
    *,
    max_rounds=DEFAULT_MAX_WITNESS_ROUNDS,
):
    """Return the final curve and explicitly report any unfinished soft objective.

    This function does not accept a trajectory: callers must independently
    verify actual collision, geometry, position bounds and source events.
    """
    return _refine_peak_witnesses(
        initial_times_s, solve, normalization, feasibility_band, max_rounds
    )
