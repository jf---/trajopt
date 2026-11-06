"""Opt-in same-scene fresh product plus actual-time peak/integral jerk pass."""

import argparse
import json
import traceback
from functools import partial
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from check_transit_sqp_callback import serializable
from fresh_actual_audit import audit_stages
from fresh_peak_refiner import refine_all_peak_transits
from fresh_product_jerk import run_verified_product
from fresh_product_scene import initialize_scene, snapshot
from fresh_product_tour import run_tour
from fresh_tour_pipeline import modeled_tour_end_secants, refine_checked_tour
from peak_transit_solver import make_peak_transit_solver
from runtime_libraries import require_loaded_libraries
from verifier_feedback.installed_native_verifier import verify

# A single explicit modeled profile for every transit in every selected tour.
PEAK_WEIGHT = 0.1
REFERENCE_TIME_S = 1.0
ABSOLUTE_MERIT_IMPROVEMENT = 1e-9
MAX_WITNESS_ROUNDS = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "output",
        "expected-wheel-module",
        "expected-sqp-library",
        "expected-ifopt-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tour", action="append", type=int, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [
            args.expected_ifopt_library.resolve(strict=True),
            args.expected_sqp_library.resolve(strict=True),
        ]
    )
    if args.output.exists():
        raise RuntimeError("output directory already exists")
    args.output.mkdir(parents=True)
    started = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup_s = perf_counter() - started
    initial = snapshot(scene)
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    single = make_peak_transit_solver(
        PEAK_WEIGHT,
        REFERENCE_TIME_S,
        absolute_merit_improvement=ABSOLUTE_MERIT_IMPROVEMENT,
        max_witness_rounds=MAX_WITNESS_ROUNDS,
    )
    transit_solver = partial(refine_all_peak_transits, solver=single)
    summaries = []

    def postpass(scene, tour, record, checked):
        result = refine_checked_tour(
            scene,
            tour,
            record,
            checked,
            raw,
            metadata,
            boundary_policy=modeled_tour_end_secants,
            transit_solver=transit_solver,
        )

        def make_spline(row):
            return native.QuinticSplineModel.build(
                row["knots_s"],
                np.asarray(row["v0"]),
                np.asarray(row["a0"]),
                np.asarray(row["v1"]),
                np.asarray(row["a1"]),
            )

        audit = audit_stages(
            result["generated_stages"], result["clock"], result["transits"], make_spline
        )
        audit["joins"] = [join._asdict() for join in audit["joins"]]
        result["actual_stage_audit"] = audit
        result["boundary_source"] = (
            "generated joint-linear secant hypothesis for solver; physical boundary derivatives unknown"
        )
        result["process_and_fixture_source"] = (
            "unchanged independently verified fresh joint-linear states and ordered events"
        )
        result["smoothing_acceptance"] = (
            "unresolved" if audit["smoothness_gate"] == "unresolved" else "not evaluated"
        )
        result["policy"] = {
            "peak_weight": PEAK_WEIGHT,
            "reference_time_s": REFERENCE_TIME_S,
            "absolute_merit_improvement": ABSOLUTE_MERIT_IMPROVEMENT,
            "max_witness_rounds": MAX_WITNESS_ROUNDS,
        }
        return result

    for tour in args.tour:
        began = perf_counter()
        try:
            result = run_verified_product(
                scene,
                tour,
                run_product=run_tour,
                verify=verify,
                postpass=postpass,
                expected_wheel_module=args.expected_wheel_module,
                expected_sqp_library=args.expected_sqp_library,
                expected_ifopt_library=args.expected_ifopt_library,
            )
            status = "sampled_candidate_with_unresolved_smoothness"
            summary = {
                "tour": tour,
                "status": status,
                "result": result,
                "elapsed_s": perf_counter() - began,
            }
        except Exception as exc:
            summary = {
                "tour": tour,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": perf_counter() - began,
            }
        if snapshot(scene) != initial:
            raise RuntimeError(f"tour {tour}: scene was not restored")
        (args.output / f"tour{tour:02d}-jerk.json").write_text(
            json.dumps(serializable(summary), indent=2, allow_nan=False) + "\n"
        )
        summaries.append(
            {key: value for key, value in summary.items() if key not in ("result", "traceback")}
        )
        print(json.dumps(summaries[-1]), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps({"scene_setup_s": setup_s, "tours": summaries}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
