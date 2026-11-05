"""Uniform soft fourth-power jerk risk over every generated Tour09 transit."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import _quintic_spline as native
import numpy as np
from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene
from risk_segments import build_risk_segments
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import make_interp_spline
from solve_duration_policy import SQP_FEASIBILITY_BAND, TransitSweepError, solve_duration_policy
from spline_speed_extrema import jerk_extrema, speed_extrema
from tour_metrics import summarize_tour
from verify_duration_policy import verify_duration_policy

# Same dimensionless risk weights and modeled reference time for every transit.
RISK_WEIGHTS = (0.0, 0.001, 0.01, 0.1)
RISK_REFERENCE_TIME_S = 1.0
# Numerical spline-parity and speed-ratio bands; neither is a physical slack.
SPLINE_PARITY_ULPS = 32768
SPEED_ROUNDOFF_ULPS = 256


class FourthGridError(RuntimeError):
    """A risk profile fails an explicit solver or verification contract."""


def inspect_transits(results, speed, baseline_curves):
    """Measure numerical polynomial speed and jerk peaks and exact quadratures."""
    peak_jerk = np.zeros(7)
    peak_speed = np.zeros(7)
    quadratic = 0.0
    fourth = 0.0
    max_node_delta = 0.0
    for result in results:
        item = result["curve"]
        nodes = np.asarray(item["nodes"])
        model = native.QuinticSplineModel.build(
            item["knots_s"],
            np.asarray(item["v0"]),
            np.asarray(item["a0"]),
            np.asarray(item["v1"]),
            np.asarray(item["a1"]),
        )
        curve = make_interp_spline(
            item["knots_s"],
            nodes,
            k=5,
            bc_type=(
                [(1, np.asarray(item["v0"])), (2, np.asarray(item["a0"]))],
                [(1, np.asarray(item["v1"])), (2, np.asarray(item["a1"]))],
            ),
        )
        velocities = speed_extrema(curve)
        jerks = jerk_extrema(curve)
        for derivative, extrema in ((1, velocities), (3, jerks)):
            times = np.asarray([x["time_s"] for x in extrema])
            actual = model.evaluate(times, nodes, derivative)[derivative]
            reference = curve(times, derivative)
            band = (
                SPLINE_PARITY_ULPS
                * np.finfo(float).eps
                * max(1.0, float(np.max(np.abs(reference))))
            )
            if np.max(np.abs(actual - reference)) > band:
                raise FourthGridError(
                    f"{item['stage']}: native and polynomial derivative models disagree"
                )
        peak_speed = np.maximum(peak_speed, [x["absolute_speed"] for x in velocities])
        peak_jerk = np.maximum(peak_jerk, [x["absolute_jerk"] for x in jerks])
        if (
            np.max(np.asarray([x["absolute_speed"] for x in velocities]) / speed)
            > 1 + SPEED_ROUNDOFF_ULPS * np.finfo(float).eps
        ):
            raise FourthGridError(f"{item['stage']}: sourced velocity limit exceeded")
        q2 = model.jerk_quadrature()
        j2 = q2.map.basis @ nodes + q2.map.offset
        quadratic += float(np.sum(np.asarray(q2.weights)[:, None] * (j2 / speed) ** 2))
        q4 = model.jerk_fourth_quadrature()
        j4 = q4.map.basis @ nodes + q4.map.offset
        fourth += float(np.sum(np.asarray(q4.weights)[:, None] * (j4 / speed) ** 4))
        if item["stage"] in baseline_curves:
            previous = np.asarray(baseline_curves[item["stage"]]["nodes"])
            if previous.shape != nodes.shape:
                raise FourthGridError("baseline knot layout differs")
            max_node_delta = max(max_node_delta, float(np.max(np.abs(nodes - previous))))
    return {
        "peak_speed_by_axis": peak_speed.tolist(),
        "peak_jerk_by_axis": peak_jerk.tolist(),
        "transit_exact_normalized_jerk_squared": quadratic,
        "transit_exact_normalized_jerk_fourth": fourth,
        "max_baseline_knot_position_difference_native_units": max_node_delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "original-worktree",
        "fresh-route",
        "verified-route",
        "artifacts",
        "fixture-policy",
        "output",
        "ifopt-library",
        "sqp-library",
        "candidate",
        "baseline-summary",
        "baseline-report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timing-note", default="Host contention not characterized")
    args = parser.parse_args()
    libraries = require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise FourthGridError("output path already exists")
    args.output.mkdir(parents=True)
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    started, cpu = perf_counter(), process_time()
    scene = initialize_scene(args.station, args.original_worktree, args.output)
    setup_s, setup_cpu_s = perf_counter() - started, process_time() - cpu
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
    segments, clock, speed = prepare_fresh(
        env,
        program,
        source,
        tours,
        home,
        names,
        raw,
        metadata,
        9,
        json.loads(args.fresh_route.read_text()),
        json.loads(args.verified_route.read_text()),
    )
    speed = np.asarray(speed, dtype=float)
    candidate = json.loads(args.candidate.read_text())
    solve_segments = build_risk_segments(segments, candidate)
    baseline = json.loads(args.baseline_summary.read_text())
    baseline_curves = {row["stage"]: row["curve"] for row in baseline["transits"]}
    manifest = {
        "scope": "One uniform 2x modeled clock and shared risk-weight grid across all 14 generated Tour09 transits. Native squared and fourth-power time-dependent jerk costs; source process/fixture geometry and clocks retained. Numerical extrema and finite collision samples; no physical/controller/manufacturing acceptance.",
        "risk_weights": RISK_WEIGHTS,
        "risk_reference_time_s": RISK_REFERENCE_TIME_S,
        "native_libraries": [str(x) for x in libraries],
        "timing_note": args.timing_note,
        "scene_setup_s": setup_s,
        "scene_setup_cpu_s": setup_cpu_s,
        "baseline_summary": str(args.baseline_summary.resolve()),
        "baseline_report": str(args.baseline_report.resolve()),
        "candidate": str(args.candidate.resolve()),
        "profiles": [],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for weight in RISK_WEIGHTS:
        folder = args.output / f"risk-{weight:g}"
        folder.mkdir()
        started, cpu = perf_counter(), process_time()
        try:
            result = solve_duration_policy(
                2.0,
                scene,
                solve_segments,
                clock,
                speed,
                folder,
                args.artifacts,
                args.fixture_policy,
                baseline_path=args.baseline_report,
                fourth_risk_weight=weight,
                risk_reference_time_s=RISK_REFERENCE_TIME_S,
            )
            solve_s, solve_cpu_s = perf_counter() - started, process_time() - cpu
            peaks = inspect_transits(result["transits"], speed, baseline_curves)
            check_start, check_cpu = perf_counter(), process_time()
            whole = verify_duration_policy(
                2.0, scene, segments, clock, speed, folder, args.artifacts, args.fixture_policy
            )
            check_s, check_cpu_s = perf_counter() - check_start, process_time() - check_cpu
            if whole["negative_contacts"] or max(whole["raw_limit_excess"]) > SQP_FEASIBILITY_BAND:
                raise FourthGridError(
                    "whole-tour sampled collision or configured joint bound failure"
                )
            joins = whole["joins"]
            profile = {
                "weight": weight,
                "status": "sampled_assembly_only",
                "solve_and_transit_checks_s": solve_s,
                "solve_and_transit_checks_cpu_s": solve_cpu_s,
                "whole_tour_checks_s": check_s,
                "whole_tour_checks_cpu_s": check_cpu_s,
                "modeled_duration_s": whole["modeled_tour_duration_s"],
                "stage_count": whole["stage_count"],
                "checked_samples": whole["checked_samples"],
                "negative_contacts": whole["negative_contacts"],
                "raw_limit_excess": whole["raw_limit_excess"],
                "max_join_gaps": {
                    key: max(x[key] for x in joins)
                    for key in ("max_position_gap", "max_velocity_jump", "max_acceleration_jump")
                },
                "whole_tour_sampled_metrics": summarize_tour(whole),
                **peaks,
                "transit_combined_objective_without_common_multiplier": peaks[
                    "transit_exact_normalized_jerk_squared"
                ]
                + weight * RISK_REFERENCE_TIME_S**4 * peaks["transit_exact_normalized_jerk_fourth"],
                "overall_accepted": False,
            }
        except (TransitSweepError, FourthGridError) as exc:
            profile = {
                "weight": weight,
                "status": "failed",
                "reason": str(exc),
                "elapsed_s": perf_counter() - started,
            }
        manifest["profiles"].append(profile)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(json.dumps(profile), flush=True)
    if not any(item["status"] == "sampled_assembly_only" for item in manifest["profiles"]):
        raise FourthGridError("all risk profiles failed; individual evidence retained")


if __name__ == "__main__":
    main()
