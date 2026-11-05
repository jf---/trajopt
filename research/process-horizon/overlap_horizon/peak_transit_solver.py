"""Native peak-epigraph transit solver with bounded actual-curve witnesses."""

import numpy as np
from native_accel_transits import TransitContractError, refine_transits
from peak_witnesses import PeakWitnessError, refine_peak_witnesses
from scipy.interpolate import make_interp_spline

# Existing SQP constraint feasibility setting, in normalized jerk units.
SQP_FEASIBILITY_BAND = 1e-4
# The original seven-knot interpolation clock is the initial witness set.
INITIAL_WITNESS_COUNT = 7


def make_peak_transit_solver(weight, reference_time_s):
    """Construct an opt-in solver; zero weight is the original native path."""

    def solve(env, info, names, segments, clock, speeds, **kwargs):
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
            result, rounds = refine_peak_witnesses(
                np.linspace(0, duration, INITIAL_WITNESS_COUNT),
                round_solve,
                normalization,
                SQP_FEASIBILITY_BAND,
            )
        except PeakWitnessError as exc:
            raise TransitContractError(f"peak witness verification failed: {exc}") from exc
        result["solution"]["peak_witness_rounds"] = rounds
        return [result["solution"]]

    return solve
