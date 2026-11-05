"""Assemble all disjoint generated probe-transit windows in one Tour09 scene."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter, process_time

from assemble_horizons import HorizonAssemblyError, assemble_horizons, eligible_windows
from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene
from run_horizon import SQP_FEASIBILITY_BAND, run
from runtime_libraries import require_loaded_libraries
from tour_metrics import summarize_tour
from verify_duration_policy import verify_duration_policy

# Floating point equality of modeled clock sums, not a physical tolerance.
CLOCK_ROUNDOFF_ULPS = 128


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
        "baseline-summary",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timing-note", default="Host contention not characterized")
    args = parser.parse_args()
    libraries = require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise HorizonAssemblyError("output path already exists")
    args.output.mkdir(parents=True)
    runtime = {
        "loaded_libraries": [str(x) for x in libraries],
        "timing_note": args.timing_note,
        "baseline_summary": str(args.baseline_summary.resolve(strict=True)),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    started, cpu = perf_counter(), process_time()
    scene = initialize_scene(args.station, args.original_worktree, args.output)
    runtime.update(
        {"scene_setup_s": perf_counter() - started, "scene_setup_cpu_s": process_time() - cpu}
    )
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
    labels = [stage["stage"] for stage in segments]
    windows = eligible_windows(labels)
    if not windows:
        raise HorizonAssemblyError("no eligible process-transit-process windows")
    reports = []
    started, cpu = perf_counter(), process_time()
    for window in windows:
        folder = args.output / window[1].replace(":", "-")
        folder.mkdir()
        reports.append(
            run(
                scene,
                segments,
                clock,
                speed,
                args.artifacts,
                args.fixture_policy,
                folder,
                window[1],
            )
        )
    runtime.update(
        {
            "window_search_and_checks_s": perf_counter() - started,
            "window_search_and_checks_cpu_s": process_time() - cpu,
        }
    )
    baseline = json.loads(args.baseline_summary.read_text())
    if float(baseline["factor"]) != 1.0:
        raise HorizonAssemblyError("baseline must use the same 1x modeled clock")
    transits, rates = assemble_horizons(labels, baseline["transits"], reports)
    folder = args.output / "global-duration-1"
    folder.mkdir()
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "factor": 1.0,
                "elapsed_s": runtime["window_search_and_checks_s"],
                "transits": transits,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (args.output / "process-rates.json").write_text(json.dumps(rates, indent=2) + "\n")
    started, cpu = perf_counter(), process_time()
    verified = verify_duration_policy(
        1.0,
        scene,
        segments,
        clock,
        speed,
        args.output,
        args.artifacts,
        args.fixture_policy,
        process_rates=rates,
    )
    runtime.update(
        {
            "whole_tour_checks_s": perf_counter() - started,
            "whole_tour_checks_cpu_s": process_time() - cpu,
        }
    )
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    if verified["negative_contacts"] or max(verified["raw_limit_excess"]) > SQP_FEASIBILITY_BAND:
        raise HorizonAssemblyError(
            "whole-tour sampled collision or configured position limit check failed; report preserved"
        )
    baseline_report = json.loads(
        (args.baseline_summary.parent / "tour09-duration-policy-assembled.json").read_text()
    )
    if [x["stage"] for x in baseline_report["stages"]] != labels:
        raise HorizonAssemblyError("baseline verified stage sequence mismatch")
    if abs(
        float(baseline_report["modeled_tour_duration_s"])
        - float(verified["modeled_tour_duration_s"])
    ) > CLOCK_ROUNDOFF_ULPS * sys.float_info.epsilon * max(
        1.0, float(verified["modeled_tour_duration_s"])
    ):
        raise HorizonAssemblyError("baseline and candidate modeled clocks differ")
    joins = verified["joins"]
    summary = {
        "scope": "Generated Tour09 with all eligible disjoint probe–transit–probe windows using the same bounded grid. Baseline transits reused elsewhere. Modeled clock, finite samples, assumed SearchL hits; no controller or manufacturing acceptance and no global optimality claim.",
        "windows": [
            {
                "stages": x["stage_sequence"],
                "selected_rates": x["selected"]["rates"],
                "selected_weighted_jerk": x["selected"]["weighted_jerk"],
                "baseline_weighted_jerk": next(
                    c["weighted_jerk"]
                    for c in x["candidates"]
                    if c["rates"] == [1.0, 1.0] and c["status"] == "valid"
                ),
                "elapsed_s": x["elapsed_s"],
            }
            for x in reports
        ],
        "stage_count": verified["stage_count"],
        "modeled_tour_duration_s": verified["modeled_tour_duration_s"],
        "checked_samples": verified["checked_samples"],
        "negative_contacts": verified["negative_contacts"],
        "raw_limit_excess": verified["raw_limit_excess"],
        "max_join_gaps": {
            key: max(x[key] for x in joins)
            for key in ("max_position_gap", "max_velocity_jump", "max_acceleration_jump")
        },
        "overall_accepted": False,
        "runtime": runtime,
        "baseline_transit_solve_and_checks_s": baseline["elapsed_s"],
        "baseline_metrics": summarize_tour(baseline_report),
        "candidate_metrics": summarize_tour(verified),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
