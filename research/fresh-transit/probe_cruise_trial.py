"""Generated probe IK with shared nullspace derivatives and modeled SearchL cruise."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage, weld_stage
from actual_stage_contract import Seconds, diagnostic_joins
from continuous_spline_jerk import build_jerk_model
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveLimitError, inspect_sampled_curve
from master_analytic_ik import arm_ik
from native_accel_transits import refine_transits
from probe_cruise_clock import search_cruise_clock
from probe_group_coordination import build_probe_group
from probe_shared_velocity import solve_shared_velocity
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import interpolate_rotations, rotation_errors
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import BPoly
from weld_tangent_leads import fit_lead

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent actual-curve check density.
SPEED_BOUND_ULPS = 32  # Binary64 allowance at a sourced axis-speed bound.


class ProbeCruiseTrialError(RuntimeError):
    """Generated IK, geometry or boundary coordination could not be established."""


def one_sided_progress(times, duration, start_rate, end_rate):
    """The existing modeled quintic phase clock, evaluated at modeled times."""
    u = np.asarray(times, dtype=float) / duration
    a = 10 - 6 * start_rate - 4 * end_rate
    b = -15 + 8 * start_rate + 7 * end_rate
    c = 6 - 3 * start_rate - 3 * end_rate
    return start_rate * u + a * u**3 + b * u**4 + c * u**5


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
        raise ProbeCruiseTrialError("verified generated candidate required")
    if args.output.exists():
        raise ProbeCruiseTrialError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    initial = snapshot(scene)
    env, _, _, _, names, opw, limits7, _, _, _, info, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("cruise_probe_nine", list(names)).getLimits().joint_limits
    )
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise ProbeCruiseTrialError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    speeds = np.asarray(speeds)
    rows, clock = candidate["generated_stages"], candidate["clock"]["stages"]
    first = next(i for i, row in enumerate(rows) if row["stage"] == f"approach:{args.probe}")
    phases, timing = rows[first : first + 4], clock[first : first + 4]
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "probe": args.probe,
        "scope": "Fresh generated process geometry and states; modeled SearchL ramp/cruise/turn and generated nullspace derivatives. No measured hit or controller timing.",
        "process": [],
        "joins": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    def jacobian(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        rotation = np.asarray(state.link_transforms["workpiece_link"].matrix)[:3, :3]
        world = np.asarray(
            env.getJointGroup(info.manipulator).calcJacobian(q[:7], "nozzle_link"), dtype=float
        )
        return np.vstack((rotation.T @ world[:3], rotation.T @ world[3:]))

    try:
        previous, _ = build_probe_group(phases, [float(t["duration_s"]) for t in timing])
        fixture = np.asarray(phases[0]["states"][0][7:], dtype=float)
        qvalues = [np.asarray(row["states"], dtype=float) for row in phases]
        if any(not np.all(q[:, 7:] == fixture) for q in qvalues):
            raise ProbeCruiseTrialError("moving fixture is unsupported")
        vbound = [np.asarray(previous[0].evaluate(Seconds(0)).velocity)[:7]]
        abound = [np.asarray(previous[0].evaluate(Seconds(0)).acceleration)[:7]]
        for j in range(3):
            left, right = qvalues[j], qvalues[j + 1]
            q0 = left[-1]
            dtl = float(timing[j]["duration_s"]) / (len(left) - 1)
            dtr = float(timing[j + 1]["duration_s"]) / (len(right) - 1)
            reference = 0.5 * (
                (left[-1, :7] - left[-2, :7]) / dtl + (right[1, :7] - right[0, :7]) / dtr
            )
            J = jacobian(q0)
            scales = np.linalg.norm(J * speeds[None, :], axis=1)
            solution = solve_shared_velocity(
                J, tuple(np.zeros(6)), tuple(reference), tuple(speeds), tuple(scales)
            )
            if solution.status != "modeled_candidate":
                raise ProbeCruiseTrialError(
                    f"{phases[j]['stage']}: shared velocity exceeds sourced speed"
                )
            velocity = np.asarray(solution.velocity)
            magnitude = float(np.max(np.abs(velocity)))
            if magnitude == 0:
                jdot = np.zeros_like(J)
            else:
                dt = (
                    np.cbrt(np.finfo(float).eps)
                    * max(1.0, float(np.max(np.abs(q0[:7]))))
                    / magnitude
                )
                qa, qb = q0.copy(), q0.copy()
                qa[:7] -= velocity * dt
                qb[:7] += velocity * dt
                jdot = (jacobian(qb) - jacobian(qa)) / (2 * dt)
            # The characteristic time changes weights only; it is not an acceleration limit.
            characteristic = min(float(timing[j]["duration_s"]), float(timing[j + 1]["duration_s"]))
            accel_weight = speeds / characteristic
            accel_scales = np.linalg.norm(J * accel_weight[None, :], axis=1)
            scaled = J * accel_weight[None, :] / accel_scales[:, None]
            delta, _, rank, _ = np.linalg.lstsq(
                scaled, -(jdot @ velocity) / accel_scales, rcond=None
            )
            if rank != 6:
                raise ProbeCruiseTrialError("shared acceleration Jacobian rank deficient")
            acceleration = accel_weight * delta
            vbound.append(velocity)
            abound.append(acceleration)
            report["joins"].append(
                {
                    "left": phases[j]["stage"],
                    "right": phases[j + 1]["stage"],
                    "velocity": velocity.tolist(),
                    "acceleration": acceleration.tolist(),
                    "zero_twist_residual": solution.task_residual,
                    "zero_twist_acceleration_residual": (
                        J @ acceleration + jdot @ velocity
                    ).tolist(),
                }
            )
        vbound.append(np.asarray(previous[-1].evaluate(previous[-1].duration).velocity)[:7])
        abound.append(np.asarray(previous[-1].evaluate(previous[-1].duration).acceleration)[:7])
        part = np.asarray(
            env.getState(dict(zip(names[7:], map(float, fixture), strict=True)))
            .link_transforms["workpiece_link"]
            .matrix
        )
        fitted, actual = [], []
        for j, (row, source, entry) in enumerate(zip(phases, qvalues, timing, strict=True)):
            source_duration = float(entry["duration_s"])
            if j in (1, 2):
                duration, progress_clock = search_cruise_clock(source_duration)

                def progress(t, fn=progress_clock):
                    return fn(t)[0]
            else:
                duration = previous[j].duration

                def progress(t, j=j, duration=duration):
                    return one_sided_progress(
                        t, duration, 1.0 if j == 0 else 0.0, 1.0 if j == 3 else 0.0
                    )

            times = np.linspace(0, duration, len(source))
            u = np.asarray(progress(times))
            first_pose, last_pose = pose(source[0]), pose(source[-1])
            rotations = interpolate_rotations(first_pose[:3, :3], last_pose[:3, :3], u)
            targets = np.tile(np.eye(4), (len(times), 1, 1))
            targets[:, :3, 3] = first_pose[:3, 3] + u[:, None] * (
                last_pose[:3, 3] - first_pose[:3, 3]
            )
            targets[:, :3, :3] = rotations
            rail = BPoly.from_derivatives(
                [0.0, float(duration)],
                [
                    [source[0, 0], vbound[j][0], abound[j][0]],
                    [source[-1, 0], vbound[j + 1][0], abound[j + 1][0]],
                ],
            )
            generated = [source[0, :7].copy()]
            branch_counts = []
            for k in range(1, len(times) - 1):
                carrier = dict(
                    zip(
                        (names[0], names[7], names[8]),
                        map(float, (rail(times[k]), *fixture)),
                        strict=True,
                    )
                )
                solved = arm_ik(
                    env,
                    opw,
                    part @ targets[k],
                    carrier,
                    tcp_link="nozzle_link",
                    limits=np.asarray(limits7)[1:],
                )
                branch_counts.append(len(solved.solutions))
                if not solved.solutions:
                    raise ProbeCruiseTrialError(
                        f"{row['stage']} sample {k}: no generated IK branch"
                    )
                choices = [np.r_[rail(times[k]), arm] for arm in solved.solutions]
                generated.append(
                    min(choices, key=lambda q: float(np.sum(((q - generated[-1]) / speeds) ** 2)))
                )
            generated.append(source[-1, :7].copy())
            points = np.asarray(generated)
            ratios = np.abs(np.diff(points, axis=0)) / np.diff(times)[:, None] / speeds
            if np.any(ratios > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps):
                raise ProbeCruiseTrialError(
                    f"{row['stage']}: generated IK step exceeds sourced speed"
                )
            curve = fit_lead(times, points, (vbound[j], abound[j]), (vbound[j + 1], abound[j + 1]))
            stage = weld_stage(row["stage"], float(duration), curve, 0.0, tuple(fixture))
            actual.append(stage)
            fitted.append(
                {
                    "curve": curve,
                    "progress": progress,
                    "source_duration": source_duration,
                    "source": source,
                    "branch_counts": branch_counts,
                    "ik_states": points.tolist(),
                }
            )
        report["process_joins"] = [x._asdict() for x in diagnostic_joins(actual)]
        for j, (stage, data) in enumerate(zip(actual, fitted, strict=True)):
            ts = np.linspace(0, stage.duration, SAMPLES)
            vals = [stage.evaluate(Seconds(float(t))) for t in ts]
            q = np.asarray([v.position for v in vals])
            v = np.asarray([v.velocity for v in vals])
            poses = np.asarray([pose(point) for point in q])
            source_first, source_last = pose(data["source"][0]), pose(data["source"][-1])
            geometry = assess_process_geometry(source_first, source_last, poses)
            u = np.asarray(data["progress"](ts))
            expected_pos = source_first[:3, 3] + u[:, None] * (
                source_last[:3, 3] - source_first[:3, 3]
            )
            expected_rot = interpolate_rotations(source_first[:3, :3], source_last[:3, :3], u)
            geometry["max_clock_position_error_m"] = float(
                np.linalg.norm(poses[:, :3, 3] - expected_pos, axis=1).max()
            )
            geometry["max_clock_orientation_error_rad"] = float(
                rotation_errors(expected_rot, poses[:, :3, :3]).max()
            )
            checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            extrema = position_extrema(data["curve"])
            excess = [
                max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                for i, e in enumerate(extrema)
            ]
            if any(x > SQP_FEASIBILITY_BAND for x in excess):
                raise CurveLimitError(f"{stage.label}: polynomial position excess")
            ratio = np.max(np.abs(v[:, :7]), axis=0) / speeds
            if np.any(ratio > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps):
                raise ProbeCruiseTrialError(f"{stage.label}: sampled sourced speed exceeded")
            feed = np.linalg.norm(np.gradient(poses[:, :3, 3], ts, axis=0), axis=1)
            jerk = np.asarray(data["curve"](ts, 3))
            cruise_feed = None
            if j in (1, 2):
                ramp = data["source_duration"] / 2
                on_cruise = (ts >= ramp) & (ts <= 2 * ramp)
                cruise_feed = [float(feed[on_cruise].min()), float(feed[on_cruise].max())]
            report["process"].append(
                {
                    "stage": stage.label,
                    "duration_s": float(stage.duration),
                    "source_duration_s": data["source_duration"],
                    "geometry": geometry,
                    "modeled_feed_range_m_s": [float(feed.min()), float(feed.max())],
                    "source_nominal_feed_m_s": float(
                        np.linalg.norm(source_last[:3, 3] - source_first[:3, 3])
                        / data["source_duration"]
                    ),
                    "sampled_cruise_feed_range_m_s": cruise_feed,
                    "sampled_peak_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                    "sampled_integrated_squared_jerk_by_axis": np.trapezoid(
                        jerk**2, ts, axis=0
                    ).tolist(),
                    "collision_and_position": checked,
                    "extrema_excess": excess,
                    "sampled_speed_ratio": ratio.tolist(),
                    "min_ik_branches": min(data["branch_counts"]),
                    "ik_states": data["ik_states"],
                }
            )
            save()
        report["status"] = "transit_refinement_in_progress"
        save()
        before, after = rows[first - 1], rows[first + 4]
        if not before["stage"].startswith("transit:") or not after["stage"].startswith("transit:"):
            raise ProbeCruiseTrialError("both generated neighboring transits required")
        points = np.vstack(
            [
                np.asarray(data["ik_states"]) if i == 0 else np.asarray(data["ik_states"])[1:]
                for i, data in enumerate(fitted)
            ]
        )
        all_states = np.column_stack((points, np.tile(fixture, (len(points), 1))))
        combined = {"stage": f"cruise-probe:{args.probe}", "states": all_states.tolist()}
        mini = [before, combined, after]
        mini_clock = {
            "stages": [
                clock[first - 1],
                {"stage": combined["stage"], "duration_s": sum(stage.duration for stage in actual)},
                clock[first + 4],
            ]
        }
        old_before = next(row for row in candidate["transits"] if row["stage"] == before["stage"])
        old_after = next(row for row in candidate["transits"] if row["stage"] == after["stage"])
        left = actual[0].evaluate(Seconds(0))
        right = actual[-1].evaluate(actual[-1].duration)
        outer_v0, outer_v1 = np.r_[old_before["v0"], 0.0, 0.0], np.r_[old_after["v1"], 0.0, 0.0]

        def boundaries(_rows, _clock, i, *, start_velocity, end_velocity):
            if i == 0:
                return start_velocity, np.asarray(left.velocity), ()
            if i == 2:
                return np.asarray(right.velocity), end_velocity, ()
            raise ProbeCruiseTrialError("unexpected transit")

        def accelerations(i):
            if i == 0:
                return np.r_[old_before["a0"], 0.0, 0.0], np.asarray(left.acceleration)
            if i == 2:
                return np.asarray(right.acceleration), np.r_[old_after["a1"], 0.0, 0.0]
            raise ProbeCruiseTrialError("unexpected transit")

        began = perf_counter()
        outcomes = refine_transits(
            env,
            info,
            names,
            mini,
            mini_clock,
            speeds,
            start_velocity=outer_v0,
            end_velocity=outer_v1,
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        report["transit_solve_s_contended"] = perf_counter() - began
        report["transits"] = []
        actual_transits = []
        for out in outcomes:
            model = native.QuinticSplineModel.build(
                out["knots_s"],
                np.asarray(out["v0"]),
                np.asarray(out["a0"]),
                np.asarray(out["v1"]),
                np.asarray(out["a1"]),
            )
            nodes, held = np.asarray(out["nodes"]), np.asarray(out["fixture"])
            ts = np.linspace(0, out["duration_s"], SAMPLES)
            pos, vel = model.evaluate(ts, nodes, 1)
            _, _, _, jerk = model.evaluate(ts, nodes, 3)
            q = np.column_stack((pos, np.tile(held, (len(pos), 1))))
            checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            curve = build_jerk_model(
                out["knots_s"],
                np.asarray(out["v0"]),
                np.asarray(out["a0"]),
                np.asarray(out["v1"]),
                np.asarray(out["a1"]),
            ).curve(nodes)
            extrema = position_extrema(curve)
            excess = [
                max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                for i, e in enumerate(extrema)
            ]
            if any(value > SQP_FEASIBILITY_BAND for value in excess):
                raise CurveLimitError(f"{out['stage']}: polynomial position excess")
            ratio = np.max(np.abs(vel), axis=0) / speeds
            if np.any(ratio > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps):
                raise ProbeCruiseTrialError(f"{out['stage']}: sampled sourced speed exceeded")
            report["transits"].append(
                {
                    "stage": out["stage"],
                    "solver_curve": out,
                    "collision_and_position": checked,
                    "extrema_excess": excess,
                    "sampled_speed_ratio": ratio.tolist(),
                    "sampled_peak_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                    "sampled_integrated_squared_jerk_by_axis": np.trapezoid(
                        jerk**2, ts, axis=0
                    ).tolist(),
                }
            )
            actual_transits.append(
                transit_stage(out["stage"], out["duration_s"], model, nodes, tuple(held))
            )
        report["all_joins"] = [
            item._asdict()
            for item in diagnostic_joins([actual_transits[0], *actual, actual_transits[1]])
        ]
        report["status"] = "sampled_probe_neighborhood_controller_and_process_acceptance_unresolved"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != initial:
        raise ProbeCruiseTrialError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "process_checked": len(report["process"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
