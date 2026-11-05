"""Bounded common-grid search over shared modeled process boundary rates."""

from itertools import product

import numpy as np


class CandidateInfeasible(RuntimeError):
    """A candidate failed an explicit constraint or finite-sample check."""


class HorizonSearchError(RuntimeError):
    """The candidate grid or a required horizon result is invalid."""


def weighted_jerk(times_s, jerk, speeds):
    """Approximate integral of squared joint jerk normalized by sourced speeds.

    Axis positions and speeds use matching native units and seconds. This is a
    numerical quadrature estimate, not a physical acceptance threshold.
    """
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(jerk, dtype=float)
    speed = np.asarray(speeds, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or values.ndim != 2
        or values.shape != (len(times), len(speed))
        or speed.ndim != 1
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or not np.isfinite(speed).all()
        or np.any(speed <= 0)
        or np.any(np.diff(times) <= 0)
    ):
        raise HorizonSearchError(
            "finite ordered times, positive speeds and matching jerk axes required"
        )
    return float(np.trapezoid(np.sum((values / speed) ** 2, axis=1), times))


# Dimensionless machine precision band for scalar line-progress arithmetic.
PROGRESS_ROUNDOFF_ULPS = 128


def require_ordered_progress(geometry):
    """Reject sampled reversal or overshoot outside a floating-point band."""
    low = float(geometry["progress_min"])
    high = float(geometry["progress_max"])
    backward = float(geometry["max_backward_progress"])
    if not np.isfinite([low, high, backward]).all():
        raise HorizonSearchError("nonfinite process progress")
    band = PROGRESS_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, abs(low), abs(high))
    if low < -band or high > 1.0 + band or backward > band:
        raise CandidateInfeasible(f"sampled process progress not ordered: {geometry}")


def search_shared_rates(grid, evaluate, checkpoint=lambda _outcomes: None):
    """Evaluate all grid pairs; retain failed candidates and select least jerk.

    The caller performs and reports every constraint check. Only explicitly
    infeasible candidates are caught; infrastructure errors propagate.
    """
    rates = np.asarray(grid, dtype=float)
    if (
        rates.ndim != 1
        or not len(rates)
        or not np.isfinite(rates).all()
        or np.any(rates <= 0)
        or len(set(rates)) != len(rates)
    ):
        raise HorizonSearchError("finite, unique, positive common candidate rates required")
    outcomes = []
    for left, right in product(rates.tolist(), repeat=2):
        try:
            result = evaluate(left, right)
        except CandidateInfeasible as exc:
            outcomes.append({"rates": [left, right], "status": "failed", "reason": str(exc)})
            checkpoint(outcomes)
            continue
        cost = float(result["weighted_jerk"])
        if not np.isfinite(cost) or cost < 0:
            raise HorizonSearchError("candidate objective must be finite and nonnegative")
        outcomes.append({"rates": [left, right], "status": "valid", **result})
        checkpoint(outcomes)
    feasible = [result for result in outcomes if result["status"] == "valid"]
    if not feasible:
        raise HorizonSearchError(f"all {len(outcomes)} horizon candidates failed: {outcomes}")
    return outcomes, min(feasible, key=lambda result: result["weighted_jerk"])
