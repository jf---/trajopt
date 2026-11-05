"""Shared soft jerk-peak weights on a generated Tour09 2x transit clock."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene
from peak_transit_solver import make_peak_transit_solver
from risk_segments import build_risk_segments
from run_fourth_grid import inspect_transits
from runtime_libraries import require_loaded_libraries
from solve_duration_policy import SQP_FEASIBILITY_BAND, TransitSweepError, solve_duration_policy
from tour_metrics import summarize_tour
from verify_duration_policy import verify_duration_policy

# One common dimensionless grid and modeled reference time for all transits.
PEAK_WEIGHTS = (0.0, 0.001, 0.01, 0.1)
REFERENCE_TIME_S = 1.0


class PeakGridError(RuntimeError):
    """A soft-peak profile violates a required solver or sampled check."""


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
        raise PeakGridError("output path already exists")
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
        "scope": "Uniform 2x modeled clock; common soft peak weight across all 14 independent transit optimizations. One free epigraph per axis and transit, not a shared whole-tour peak variable. Actual quintic peak witnesses are refined with a finite budget and SQP tolerance; process/fixture geometry and clocks are retained. Finite sampled checks, no physical or manufacturing acceptance.",
        "peak_weights": PEAK_WEIGHTS,
        "reference_time_s": REFERENCE_TIME_S,
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
    for weight in PEAK_WEIGHTS:
        folder = args.output / f"peak-{weight:g}"
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
                transit_solver=make_peak_transit_solver(weight, REFERENCE_TIME_S),
            )
            solve_s, solve_cpu_s = perf_counter() - started, process_time() - cpu
            peaks = inspect_transits(result["transits"], speed, baseline_curves)
            check_start, check_cpu = perf_counter(), process_time()
            whole = verify_duration_policy(
                2.0, scene, segments, clock, speed, folder, args.artifacts, args.fixture_policy
            )
            check_s, check_cpu_s = perf_counter() - check_start, process_time() - check_cpu
            if whole["negative_contacts"] or max(whole["raw_limit_excess"]) > SQP_FEASIBILITY_BAND:
                raise PeakGridError(
                    "whole-tour sampled collision or configured joint bound failure"
                )
            joins = whole["joins"]
            rows = []
            for item in result["transits"]:
                curve = item["curve"]
                rows.append(
                    {
                        "stage": item["stage"],
                        "epigraph": curve.get("peak_epigraph"),
                        "witness_rounds": curve.get("peak_witness_rounds", []),
                    }
                )
            risk_cost = sum(
                float(np.dot(row["epigraph"], row["epigraph"]))
                for row in rows
                if row["epigraph"] is not None
            )
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
                "peak_epigraphs_and_witness_rounds": rows,
                "transit_combined_objective_without_common_multiplier": peaks[
                    "transit_exact_normalized_jerk_squared"
                ]
                + weight / REFERENCE_TIME_S**3 * risk_cost,
                "overall_accepted": False,
            }
        except (TransitSweepError, PeakGridError) as exc:
            profile = {
                "weight": weight,
                "status": "failed",
                "reason": str(exc),
                "elapsed_s": perf_counter() - started,
            }
        manifest["profiles"].append(profile)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {k: v for k, v in profile.items() if k != "peak_epigraphs_and_witness_rounds"}
            ),
            flush=True,
        )
    if not any(item["status"] == "sampled_assembly_only" for item in manifest["profiles"]):
        raise PeakGridError("all peak profiles failed; individual evidence retained")


if __name__ == "__main__":
    main()
