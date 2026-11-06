"""Fresh tangent-lead, unchanged-bead feasibility experiment in one scene."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from actual_stage_adapters import weld_stage
from actual_stage_contract import Seconds, diagnostic_joins
from check_joint_motion_graph import LEAD_LENGTH_M
from cubic_position_extrema import cubic_position_extrema
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve
from master_analytic_ik import arm_ik
from process_curve_representation import build_process_curve
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import interpolate_rotations, rotation_errors
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from weld_tangent_leads import fit_lead, tangent_lead_targets

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Same independent density as the existing spline checks, not continuous proof.
SAMPLES = 1025
SPEED_BOUND_ULPS = 32


class WeldTrialError(RuntimeError):
    """The selected fresh weld cannot support the stated lead experiment."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--weld-stage", required=True)
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
        raise WeldTrialError("independently verified generated candidate required")
    if args.output.exists():
        raise WeldTrialError("output already exists")
    args.output.mkdir(parents=True)
    start = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - start
    initial = snapshot(scene)
    env, _, _, _, names, opw, limits7, _, _, _, _, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("tangent_lead_nine", list(names)).getLimits().joint_limits
    )
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise WeldTrialError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    speeds = np.asarray(speeds)

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    rows, timings = candidate["generated_stages"], candidate["clock"]["stages"]
    index = next(i for i, row in enumerate(rows) if row["stage"] == args.weld_stage)
    report = {
        "scope": "One generated bead and tangent-aligned regenerated leads; fixed generated rail; finite sampling; no process or controller acceptance.",
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "stages": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        if not 0 < index < len(rows) - 1 or not all(
            rows[j]["stage"].startswith("weld:") for j in (index - 1, index, index + 1)
        ):
            raise WeldTrialError("lead, bead and lead consecutive stages required")
        if (
            index < 2
            or index + 2 >= len(rows)
            or not rows[index - 2]["stage"].startswith("transit:")
            or not rows[index + 2]["stage"].startswith("transit:")
        ):
            raise WeldTrialError("single-edge bead between lead-adjacent transits required")
        bead = np.asarray(rows[index]["states"], dtype=float)
        if not np.all(bead[:, 0] == bead[0, 0]) or not np.all(bead[:, 7:] == bead[0, 7:]):
            raise WeldTrialError("the experiment requires generated fixed rail and fixture")
        fixture = bead[0, 7:]
        bead_time = float(timings[index]["duration_s"])
        bead_stage, bead_curve = build_process_curve(args.weld_stage, bead, bead_time)
        first, last = pose(bead[0]), pose(bead[-1])
        before_targets, after_targets = tangent_lead_targets(
            first, last, float(LEAD_LENGTH_M), len(bead)
        )
        carrier = dict(
            zip((names[0], names[7], names[8]), map(float, (bead[0, 0], *fixture)), strict=True)
        )
        part = np.asarray(env.getState(carrier).link_transforms["workpiece_link"].matrix)
        ik_counts, ik_refused = [], []

        def generated_lead(targets, anchor, reverse, label):
            order = list(reversed(targets[:-1])) if reverse else list(targets[1:])
            previous = anchor.copy()
            result = [previous]
            for position, target in enumerate(order, start=1):
                answer = arm_ik(
                    env,
                    opw,
                    part @ target,
                    carrier,
                    tcp_link="nozzle_link",
                    limits=np.asarray(limits7)[1:],
                )
                ik_counts.append(len(answer.solutions))
                ik_refused.append(len(answer.refused))
                if not answer.solutions:
                    raise WeldTrialError(
                        f"{label} sample {position}: no valid IK at fixed generated rail"
                    )
                options = [np.r_[bead[0, 0], solution] for solution in answer.solutions]
                previous = min(options, key=lambda q: float(np.sum(((q - previous) / speeds) ** 2)))
                result.append(previous)
            return np.asarray(list(reversed(result)) if reverse else result)

        begin = perf_counter()
        q_in = generated_lead(before_targets, bead[0, :7], True, rows[index - 1]["stage"])
        q_out = generated_lead(after_targets, bead[-1, :7], False, rows[index + 1]["stage"])
        report["ik_s_contended"] = perf_counter() - begin
        report["ik_queries"] = len(ik_counts)
        report["min_ik_branches"] = min(ik_counts)
        report["refused_ik_solutions"] = sum(ik_refused)
        report["max_arm_branch_step_rad"] = float(
            max(np.max(np.abs(np.diff(q[:, 1:], axis=0))) for q in (q_in, q_out))
        )
        duration_in, duration_out = (
            float(timings[index - 1]["duration_s"]),
            float(timings[index + 1]["duration_s"]),
        )
        t_in, t_out = (
            np.linspace(0, duration_in, len(q_in)),
            np.linspace(0, duration_out, len(q_out)),
        )
        raw_in, raw_out = CubicSpline(t_in, q_in, axis=0), CubicSpline(t_out, q_out, axis=0)
        curve_in = fit_lead(
            t_in, q_in, (raw_in(0, 1), raw_in(0, 2)), (bead_curve(0, 1), bead_curve(0, 2))
        )
        curve_out = fit_lead(
            t_out,
            q_out,
            (bead_curve(bead_time, 1), bead_curve(bead_time, 2)),
            (raw_out(duration_out, 1), raw_out(duration_out, 2)),
        )
        stage_in = weld_stage(rows[index - 1]["stage"], duration_in, curve_in, 0.0, tuple(fixture))
        stage_out = weld_stage(
            rows[index + 1]["stage"], duration_out, curve_out, 0.0, tuple(fixture)
        )
        stages, curves = [stage_in, bead_stage, stage_out], [curve_in, bead_curve, curve_out]
        report["joins"] = [item._asdict() for item in diagnostic_joins(stages)]
        report["generated_lead_samples"] = {"lead_in": q_in.tolist(), "lead_out": q_out.tolist()}
        for stage, curve in zip(stages, curves, strict=True):
            times = np.linspace(0, stage.duration, SAMPLES)
            states = [stage.evaluate(Seconds(float(t))) for t in times]
            q = np.asarray([state.position for state in states])
            vel = np.asarray([state.velocity for state in states])
            poses = np.asarray([pose(state) for state in q])
            geometry = assess_process_geometry(poses[0], poses[-1], poses)
            progress = times / stage.duration
            target_position = poses[0, :3, 3] + progress[:, None] * (
                poses[-1, :3, 3] - poses[0, :3, 3]
            )
            target_rotation = interpolate_rotations(poses[0, :3, :3], poses[-1, :3, :3], progress)
            geometry["max_clock_position_error_m"] = float(
                np.linalg.norm(poses[:, :3, 3] - target_position, axis=1).max()
            )
            geometry["max_clock_orientation_error_rad"] = float(
                rotation_errors(target_rotation, poses[:, :3, :3]).max()
            )
            feed = np.linalg.norm(np.gradient(poses[:, :3, 3], times, axis=0), axis=1)
            try:
                checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            except (CurveCollisionError, CurveLimitError) as exc:
                checked = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            extrema = (
                cubic_position_extrema(curve) if stage is bead_stage else position_extrema(curve)
            )
            excess = [
                max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                for i, e in enumerate(extrema)
            ]
            ratio = np.max(np.abs(vel[:, :7]), axis=0) / speeds
            report["stages"].append(
                {
                    "stage": stage.label,
                    "duration_s": float(stage.duration),
                    "geometry": geometry,
                    "modeled_feed_range_m_s": [float(feed.min()), float(feed.max())],
                    "nominal_modeled_feed_m_s": float(
                        np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]) / stage.duration
                    ),
                    "sampled_peak_jerk_by_joint": np.max(np.abs(curve(times, 3)), axis=0).tolist(),
                    "sampled_speed_ratio": ratio.tolist(),
                    "sampled_speeds_pass": bool(
                        np.all(ratio <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps)
                    ),
                    "collision_and_position": checked,
                    "extrema_excess": excess,
                }
            )
        check_failed = any(
            row["collision_and_position"].get("status") == "failed"
            or not row["sampled_speeds_pass"]
            or any(excess > SQP_FEASIBILITY_BAND for excess in row["extrema_excess"])
            for row in report["stages"]
        )
        report["status"] = (
            "failed_actual_curve_checks" if check_failed else "sampled_lead_bead_diagnostic_only"
        )
        report["unresolved"] = [
            "outer lead-transit joins",
            "lead approach admissibility",
            "physical timing",
            "manufacturing tolerance",
            "continuous collision",
        ]
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != initial:
        raise WeldTrialError("scene state changed")
    print(json.dumps({"status": report["status"], "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
