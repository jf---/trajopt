"""Generate all probes and shared transits for a fresh multi-cluster tour."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from actual_stage_contract import Seconds
from event_aware_spline_boundaries import require_fixed_fixture_boundary_velocities
from fixture_easing_policy import ease_fixture_chains
from fresh_product_scene import initialize_scene, snapshot
from native_accel_transits import refine_transits
from probe_cruise_generate import generate_cruise_group
from runtime_libraries import require_loaded_libraries
from transit_neighbor_map import transit_neighbor_map
from verified_linear_stages import actual_linear_stage


class MultiProbeTrialError(RuntimeError):
    """A source tour, event, fixture or actual boundary is unsupported."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    saved = json.loads(args.input.read_text())["result"]
    candidate = saved["jerk"]
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
    ):
        raise MultiProbeTrialError("independently verified generated candidate required")
    if args.output.exists():
        raise MultiProbeTrialError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    original = snapshot(scene)
    env, _, _, _, names, opw, limits7, _, _, _, info, _ = scene["inputs"]
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise MultiProbeTrialError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    rows = [dict(row) for row in candidate["generated_stages"]]
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "scope": "Fresh generated process, original weld interpolation and ordered fixture paths; generated probes and transits await one independent actual-curve verifier.",
        "groups": [],
        "transits": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        clock, fixture_stages = ease_fixture_chains(rows, candidate["clock"])
        timing = clock["stages"]
        actual = {}
        for i, row in enumerate(rows):
            if not row["stage"].startswith("approach:"):
                continue
            if i + 4 > len(rows):
                raise MultiProbeTrialError("incomplete generated probe")
            group, evidence = generate_cruise_group(
                env, opw, info, names, limits7, speeds, rows[i : i + 4], timing[i : i + 4]
            )
            for j, stage in enumerate(group, start=i):
                actual[j] = stage
                timing[j]["duration_s"] = float(stage.duration)
                generated = np.asarray(evidence["process"][j - i]["ik_states"])
                fixture = np.asarray(rows[j]["states"], dtype=float)[0, 7:]
                rows[j] = {
                    "stage": stage.label,
                    "states": np.column_stack(
                        (generated, np.tile(fixture, (len(generated), 1)))
                    ).tolist(),
                }
            report["groups"].append(
                {
                    "probe": row["stage"].split(":")[1],
                    "stages": [stage.label for stage in group],
                    **evidence,
                }
            )
            save()
        if not report["groups"]:
            raise MultiProbeTrialError("no generated probes")
        for i, row in enumerate(rows):
            if row["stage"].startswith("transit:") or i in actual:
                continue
            if row["stage"] in fixture_stages:
                actual[i] = fixture_stages[row["stage"]]
            else:
                actual[i] = actual_linear_stage(
                    row["stage"], row["states"], float(timing[i]["duration_s"])
                )
        old = {row["stage"]: row for row in candidate["transits"]}
        if not rows[0]["stage"].startswith("transit:") or not rows[-1]["stage"].startswith(
            "transit:"
        ):
            raise MultiProbeTrialError("modeled outer transit boundaries required")
        if rows[0]["stage"] not in old or rows[-1]["stage"] not in old:
            raise MultiProbeTrialError("prior generated outer derivatives required")
        outer_v0, outer_v1 = (
            np.r_[old[rows[0]["stage"]]["v0"], 0.0, 0.0],
            np.r_[old[rows[-1]["stage"]]["v1"], 0.0, 0.0],
        )
        outer_a0, outer_a1 = (
            np.r_[old[rows[0]["stage"]]["a0"], 0.0, 0.0],
            np.r_[old[rows[-1]["stage"]]["a1"], 0.0, 0.0],
        )

        # Rebuild node timestamps after changing the process-stage durations.
        clock, _ = ease_fixture_chains(rows, clock)
        timing = clock["stages"]

        neighbors = transit_neighbor_map(rows, timing, actual)

        def boundaries(_rows, _timing, index, *, start_velocity, end_velocity):
            left, right, events = neighbors[index]
            v0 = start_velocity if left is None else left.evaluate(left.duration).velocity
            v1 = end_velocity if right is None else right.evaluate(Seconds(0)).velocity
            if v0 is None or v1 is None:
                raise MultiProbeTrialError(f"{rows[index]['stage']}: unknown neighboring velocity")
            require_fixed_fixture_boundary_velocities(v0, v1)
            return np.asarray(v0), np.asarray(v1), events

        def accelerations(index):
            left, right, _ = neighbors[index]
            a0 = outer_a0 if left is None else left.evaluate(left.duration).acceleration
            a1 = outer_a1 if right is None else right.evaluate(Seconds(0)).acceleration
            if a0 is None or a1 is None:
                raise MultiProbeTrialError(
                    f"{rows[index]['stage']}: unknown neighboring acceleration"
                )
            return np.asarray(a0), np.asarray(a1)

        begin = perf_counter()
        outcomes = refine_transits(
            env,
            info,
            names,
            rows,
            clock,
            speeds,
            start_velocity=outer_v0,
            end_velocity=outer_v1,
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        report["solve_s_contended"] = perf_counter() - begin
        report["transits"] = outcomes
        report["clock"] = clock
        report["fixture_stages"] = list(fixture_stages)
        report["status"] = "optimization_completed_awaiting_actual_curve_verification"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != original:
        raise MultiProbeTrialError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "groups": len(report["groups"]),
                "transits": len(report["transits"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
