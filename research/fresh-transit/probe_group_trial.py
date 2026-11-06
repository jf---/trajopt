"""Check one modeled probe and optimize its two generated neighboring transits."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage
from actual_stage_contract import Seconds, diagnostic_joins
from cubic_position_extrema import cubic_position_extrema
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveLimitError, inspect_sampled_curve
from native_accel_transits import refine_transits
from probe_group_coordination import build_probe_group
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Same independent density as transit checks; collision remains sampled.
SAMPLES = 1025
SPEED_BOUND_ULPS = 32


class ProbeTrialError(RuntimeError):
    """The generated candidate or scene cannot support this diagnostic."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--probe", type=int, required=True)
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
        raise ProbeTrialError("independently verified fresh candidate required")
    if args.output.exists():
        raise ProbeTrialError("output already exists")
    args.output.mkdir(parents=True)
    start = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - start
    scene_initial = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(env.getJointGroup("probe_group_nine", list(names)).getLimits().joint_limits)
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise ProbeTrialError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    rows, clock = candidate["generated_stages"], candidate["clock"]["stages"]
    index = next(i for i, row in enumerate(rows) if row["stage"] == f"approach:{args.probe}")
    subset = rows[index : index + 4]
    timings = clock[index : index + 4]
    stages, curves = build_probe_group(subset, [float(item["duration_s"]) for item in timings])
    report = {
        "status": "process_checks_in_progress",
        "scene_setup_s_contended": setup,
        "scope": "Fixed generated joint paths; explicit modeled stops at probe joins; no controller or manufacturing acceptance.",
        "modeled_stops": [
            f"{a.label} -> {b.label}" for a, b in zip(stages[:-1], stages[1:], strict=True)
        ],
        "process": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        for row, timing, stage, curve in zip(subset, timings, stages, curves, strict=True):
            times = np.linspace(0, stage.duration, SAMPLES)
            values = [stage.evaluate(Seconds(float(t))) for t in times]
            q = np.asarray([value.position for value in values])
            v = np.asarray([value.velocity for value in values])
            checks = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            extrema = cubic_position_extrema(curve)
            excess = [
                max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                for i, e in enumerate(extrema)
            ]
            if any(x > SQP_FEASIBILITY_BAND for x in excess):
                raise CurveLimitError(f"{stage.label}: cubic extrema exceed configured band")
            poses = np.asarray([pose(state) for state in q])
            geometry = assess_process_geometry(poses[0], poses[-1], poses)
            feed = np.linalg.norm(np.gradient(poses[:, :3, 3], times, axis=0), axis=1)
            source_feed = float(
                np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]) / float(timing["duration_s"])
            )
            ratio = np.max(np.abs(v[:, :7]), axis=0) / speeds
            report["process"].append(
                {
                    "stage": stage.label,
                    "duration_s": float(stage.duration),
                    "source_duration_s": timing["duration_s"],
                    "geometry": geometry,
                    "modeled_feed_range_m_s": [float(feed.min()), float(feed.max())],
                    "original_uniform_feed_m_s": source_feed,
                    "sampled_speed_ratio": ratio.tolist(),
                    "sampled_speeds_pass": bool(
                        np.all(ratio <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps)
                    ),
                    "collision_and_position": checks,
                    "extrema_excess": excess,
                }
            )
        report["process_joins"] = [item._asdict() for item in diagnostic_joins(stages)]
        report["status"] = "transit_refinement_in_progress"
        save()
        before, after = rows[index - 1], rows[index + 4]
        if not before["stage"].startswith("transit:") or not after["stage"].startswith("transit:"):
            raise ProbeTrialError("probe needs two adjacent transits")
        combined = {
            "stage": f"probe:{args.probe}",
            "states": [
                q
                for i, row in enumerate(subset)
                for q in (row["states"] if i == 0 else row["states"][1:])
            ],
        }
        mini = [before, combined, after]
        miniature_clock = {
            "stages": [
                clock[index - 1],
                {"stage": combined["stage"], "duration_s": sum(stage.duration for stage in stages)},
                clock[index + 4],
            ]
        }
        old_before = next(x for x in candidate["transits"] if x["stage"] == before["stage"])
        old_after = next(x for x in candidate["transits"] if x["stage"] == after["stage"])
        process_start = stages[0].evaluate(Seconds(0))
        process_end = stages[-1].evaluate(stages[-1].duration)
        left_outer = np.r_[old_before["v0"], 0.0, 0.0]
        right_outer = np.r_[old_after["v1"], 0.0, 0.0]

        def boundaries(_rows, _clock, i, *, start_velocity, end_velocity):
            if i == 0:
                return start_velocity, np.asarray(process_start.velocity), ()
            if i == 2:
                return np.asarray(process_end.velocity), end_velocity, ()
            raise ProbeTrialError("unexpected transit")

        def accelerations(i):
            if i == 0:
                return np.r_[old_before["a0"], 0.0, 0.0], np.asarray(process_start.acceleration)
            if i == 2:
                return np.asarray(process_end.acceleration), np.r_[old_after["a1"], 0.0, 0.0]
            raise ProbeTrialError("unexpected transit")

        start = perf_counter()
        outcomes = refine_transits(
            env,
            info,
            names,
            mini,
            miniature_clock,
            speeds,
            start_velocity=left_outer,
            end_velocity=right_outer,
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        report["transit_solve_s_contended"] = perf_counter() - start
        actual_transits = []
        report["transits"] = []
        for out in outcomes:
            model = native.QuinticSplineModel.build(
                out["knots_s"],
                np.asarray(out["v0"]),
                np.asarray(out["a0"]),
                np.asarray(out["v1"]),
                np.asarray(out["a1"]),
            )
            nodes, fixture = np.asarray(out["nodes"]), np.asarray(out["fixture"])
            ts = np.linspace(0, out["duration_s"], SAMPLES)
            pos, vel = model.evaluate(ts, nodes, 1)
            q = np.column_stack((pos, np.tile(fixture, (len(pos), 1))))
            checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            from continuous_spline_jerk import build_jerk_model

            polynomial = build_jerk_model(
                out["knots_s"],
                np.asarray(out["v0"]),
                np.asarray(out["a0"]),
                np.asarray(out["v1"]),
                np.asarray(out["a1"]),
            ).curve(nodes)
            extrema = position_extrema(polynomial)
            excess = [
                max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                for i, e in enumerate(extrema)
            ]
            if any(x > SQP_FEASIBILITY_BAND for x in excess):
                raise CurveLimitError(f"{out['stage']}: extrema exceed configured band")
            ratio = np.max(np.abs(vel), axis=0) / speeds
            report["transits"].append(
                {
                    "stage": out["stage"],
                    "status": out["status"],
                    "solver_curve": out,
                    "sampled_speed_ratio": ratio.tolist(),
                    "sampled_speeds_pass": bool(
                        np.all(ratio <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps)
                    ),
                    "collision_and_position": checked,
                    "extrema_excess": excess,
                }
            )
            actual_transits.append(
                transit_stage(out["stage"], out["duration_s"], model, nodes, tuple(fixture))
            )
        report["all_joins"] = [
            item._asdict()
            for item in diagnostic_joins([actual_transits[0], *stages, actual_transits[1]])
        ]
        speeds_pass = all(
            row["sampled_speeds_pass"] for row in report["process"] + report["transits"]
        )
        report["status"] = (
            "sampled_neighborhood_diagnostic_only"
            if speeds_pass
            else "failed_sampled_sourced_speed"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != scene_initial:
        raise ProbeTrialError("scene state changed")
    print(json.dumps({"status": report["status"], "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
