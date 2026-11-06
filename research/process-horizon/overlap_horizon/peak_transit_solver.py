"""Native peak-epigraph transit solver with bounded actual-curve witnesses."""

import numpy as np
from native_accel_transits import TransitContractError, refine_transits
from peak_witnesses import (
    DEFAULT_MAX_WITNESS_ROUNDS,
    PeakWitnessError,
    refine_peak_witnesses,
    refine_peak_witnesses_bounded,
)
from scipy.interpolate import make_interp_spline

# Existing SQP constraint feasibility setting, in normalized jerk units.
SQP_FEASIBILITY_BAND = 1e-4
# The original seven-knot interpolation clock is the initial witness set.
INITIAL_WITNESS_COUNT = 7


def make_peak_transit_solver(
    weight,
    reference_time_s,
    *,
    absolute_merit_improvement=None,
    max_witness_rounds=DEFAULT_MAX_WITNESS_ROUNDS,
):
    """Construct a strict solver requiring every peak witness to satisfy its band."""
    return _make_peak_transit_solver(
        weight, reference_time_s, absolute_merit_improvement, max_witness_rounds, False
    )


def make_bounded_peak_transit_solver(
    weight,
    reference_time_s,
    *,
    absolute_merit_improvement=None,
    max_witness_rounds=DEFAULT_MAX_WITNESS_ROUNDS,
):
    """Retain a budgeted result while explicitly reporting incomplete objective refinement."""
    return _make_peak_transit_solver(
        weight, reference_time_s, absolute_merit_improvement, max_witness_rounds, True
    )


def _make_peak_transit_solver(
    weight, reference_time_s, absolute_merit_improvement, max_witness_rounds, bounded
):
    if (
        isinstance(max_witness_rounds, bool)
        or not isinstance(max_witness_rounds, int)
        or max_witness_rounds <= 0
    ):
        raise TransitContractError("positive integer witness budget required")

    def solve(env, info, names, segments, clock, speeds, **kwargs):
        if absolute_merit_improvement is not None:
            kwargs["absolute_merit_improvement"] = absolute_merit_improvement
        if weight == 0:
            return refine_transits(env, info, names, segments, clock, speeds, **kwargs)
        selected = [
            index for index, item in enumerate(segments) if item["stage"].startswith("transit:")
        ]
        if len(selected) != 1:
            raise TransitContractError("peak witness refinement requires one transit")
        duration = float(clock["stages"][selected[0]]["duration_s"])
        normalization = reference_time_s**2 / np.asarray(speeds, dtype=float)

        def round_solve(witnesses, previous):
            seed = None if previous is None else previous["solution"]["nodes"]
            accelerations = (
                None
                if previous is None
                else (previous["solution"]["a0"], previous["solution"]["a1"])
            )
            result = refine_transits(
                env,
                info,
                names,
                segments,
                clock,
                speeds,
                **kwargs,
                peak_risk_weight=weight,
                peak_reference_time_s=reference_time_s,
                peak_witness_times_s=witnesses,
                seed_positions=seed,
                seed_accelerations=accelerations,
            )
            if len(result) != 1:
                raise TransitContractError("peak refinement expected one result")
            solution = result[0]
            nodes = np.asarray(solution["nodes"])
            curve = make_interp_spline(
                solution["knots_s"],
                nodes,
                k=5,
                bc_type=(
                    [(1, np.asarray(solution["v0"])), (2, np.asarray(solution["a0"]))],
                    [(1, np.asarray(solution["v1"])), (2, np.asarray(solution["a1"]))],
                ),
            )
            return {
                "curve": curve,
                "peak_epigraph": solution["peak_epigraph"],
                "solution": solution,
            }

        try:
            if bounded:
                result, rounds, witness_status = refine_peak_witnesses_bounded(
                    np.linspace(0, duration, INITIAL_WITNESS_COUNT),
                    round_solve,
                    normalization,
                    SQP_FEASIBILITY_BAND,
                    max_rounds=max_witness_rounds,
                )
            else:
                result, rounds = refine_peak_witnesses(
                    np.linspace(0, duration, INITIAL_WITNESS_COUNT),
                    round_solve,
                    normalization,
                    SQP_FEASIBILITY_BAND,
                    max_rounds=max_witness_rounds,
                )
                witness_status = "satisfied"
        except PeakWitnessError as exc:
            raise TransitContractError(f"peak witness verification failed: {exc}") from exc
        result["solution"]["peak_witness_rounds"] = rounds
        result["solution"]["peak_witness_status"] = witness_status
        return [result["solution"]]

    return solve
