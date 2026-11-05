"""Isolated generated probe research experiment."""

import json
from time import perf_counter

import _quintic_spline as native
import numpy as np
from all24_spline_scene import native_log
from free_space_transit import _restore_scene
from fresh_scene import snapshot
from native_accel_transits import refine_transits
from rotation_targets import interpolate_rotations, rotation_errors
from scipy.interpolate import CubicSpline

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters


def run_probe_neighbors(scene, segments, clock, speed, output):
    before = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]

    raw = json.loads((output / "ik-progress-report.json").read_text())
    retime = json.loads((output / "ik-progress-retime.json").read_text())
    labels = ["approach:0", "search:0", "retract:0", "withdraw:0"]
    first = next(i for i, s in enumerate(segments) if s["stage"] == labels[0])
    stages = segments[first : first + 4]
    before_stage = segments[first - 1]
    after_stage = segments[first + 4]
    later = segments[first + 5]
    edges = np.asarray(raw["edges_s"])
    fixture = np.asarray(raw["fixture"])
    qs = [np.asarray(x) for x in raw["stage_states"]]
    # Exact generated join configurations are canonical; this changes only FK roundoff.
    start_original = np.asarray(stages[0]["states"][0])[:7]
    end_original = np.asarray(stages[-1]["states"][-1])[:7]
    normalization_max = float(
        max(np.max(np.abs(qs[0][0] - start_original)), np.max(np.abs(qs[-1][-1] - end_original)))
    )
    qs[0][0] = start_original
    qs[-1][-1] = end_original
    curves = [CubicSpline(np.linspace(0, 1, len(q)), q, axis=0) for q in qs]
    velocity0 = np.asarray(retime["first_process_velocity"])
    velocity1 = np.asarray(retime["last_process_velocity"])
    accel0 = np.asarray(retime["outer_accelerations"][0])
    accel1 = np.asarray(retime["outer_accelerations"][1])
    zero9 = np.zeros(9)
    parts = []
    for q in qs:
        parts.extend(
            np.column_stack((q, np.tile(fixture, (len(q), 1))))
            if not parts
            else np.column_stack((q, np.tile(fixture, (len(q), 1))))[1:]
        )
    combined = {"stage": "ik-process:0", "states": np.asarray(parts).tolist()}
    mini = [before_stage, combined, after_stage, later]
    times = [
        clock["stages"][first - 1],
        {"stage": "ik-process:0", "duration_s": float(edges[-1])},
        clock["stages"][first + 4],
        clock["stages"][first + 5],
    ]

    def slope(stage, timing, side):
        q = np.asarray(stage["states"])
        duration = float(timing["duration_s"])
        return ((q[1] - q[0]) if side == "start" else (q[-1] - q[-2])) * (len(q) - 1) / duration

    start_velocity = slope(before_stage, times[0], "start")
    next_velocity = slope(later, times[3], "start")
    p0 = np.r_[velocity0, 0.0, 0.0]
    p1 = np.r_[velocity1, 0.0, 0.0]
    a0 = np.r_[accel0, 0.0, 0.0]
    a1 = np.r_[accel1, 0.0, 0.0]

    def boundary_resolver(stages_, clock_, index, *, start_velocity, end_velocity):
        if index == 0:
            return start_velocity, p0, ()
        if index == 2:
            return p1, end_velocity, ()
        raise RuntimeError("unexpected transit in selected probe neighborhood")

    def acceleration_resolver(index):
        if index == 0:
            return zero9, a0
        if index == 2:
            return a1, zero9
        raise RuntimeError("unexpected transit acceleration request")

    started = perf_counter()
    with native_log(output / "probe-neighbors-native.log"):
        results = refine_transits(
            env,
            info,
            names,
            mini,
            {"stages": times},
            speed,
            start_velocity=start_velocity,
            end_velocity=next_velocity,
            boundary_resolver=boundary_resolver,
            acceleration_resolver=acceleration_resolver,
        )
    solve_s = perf_counter() - started
    (output / "probe-neighbors-transits.json").write_text(
        json.dumps({"refine_transits_s": solve_s, "transits": results}, indent=2) + "\n"
    )
    initial = {str(k): float(v) for k, v in env.getState().joints.items()}
    revision = int(env.getRevision())
    try:
        env.setState(dict(zip(names[7:], map(float, fixture), strict=True)))
        checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
        limits = np.asarray(
            env.getJointGroup("probe-neighbor-nine", list(names)).getLimits().joint_limits
        )
        samples = []
        piece_labels = []
        joints = []
        geometric = []

        def pose(q):
            state = env.getState(
                {
                    **dict(zip(names[7:], map(float, fixture), strict=True)),
                    **dict(zip(names[:7], map(float, q), strict=True)),
                }
            )
            return np.linalg.solve(
                np.asarray(state.link_transforms["workpiece_link"].matrix),
                np.asarray(state.link_transforms["nozzle_link"].matrix),
            )

        for result in results:
            model = native.QuinticSplineModel.build(
                result["knots_s"],
                np.asarray(result["v0"]),
                np.asarray(result["a0"]),
                np.asarray(result["v1"]),
                np.asarray(result["a1"]),
            )
            ts = np.linspace(0, result["duration_s"], 1025)
            q, v = model.evaluate(ts, np.asarray(result["nodes"]), 1)
            _, _, a = model.evaluate(ts, np.asarray(result["nodes"]), 2)
            samples.extend(np.column_stack((q, np.tile(fixture, (len(q), 1)))))
            piece_labels.extend([result["stage"]] * len(q))
            joints.append(
                {
                    "stage": result["stage"],
                    "start_position": q[0].tolist(),
                    "end_position": q[-1].tolist(),
                    "start_velocity": v[0].tolist(),
                    "end_velocity": v[-1].tolist(),
                    "start_acceleration": a[0].tolist(),
                    "end_acceleration": a[-1].tolist(),
                    "max_speed_ratio": float(np.max(np.abs(v) / np.asarray(speed))),
                }
            )
        for i, (stage, curve) in enumerate(zip(stages, curves, strict=True)):
            v0 = 1.0 if i == 0 else 0.0
            v1 = 1.0 if i == 3 else 0.0
            tau = np.linspace(0, 1, 257)
            a3 = 10 - 6 * v0 - 4 * v1
            a4 = -15 + 8 * v0 + 7 * v1
            a5 = 6 - 3 * v0 - 3 * v1
            s = v0 * tau + a3 * tau**3 + a4 * tau**4 + a5 * tau**5
            s = np.clip(s, 0, 1)
            q = curve(s)
            actual = np.asarray([pose(x) for x in q])
            a = pose(np.asarray(stage["states"][0])[:7])
            b = pose(np.asarray(stage["states"][-1])[:7])
            target = a[:3, 3] + s[:, None] * (b[:3, 3] - a[:3, 3])
            desired = interpolate_rotations(a[:3, :3], b[:3, :3], s)
            pe = np.linalg.norm(actual[:, :3, 3] - target, axis=1)
            oe = rotation_errors(desired, actual[:, :3, :3])
            axis = b[:3, 3] - a[:3, 3]
            progress = (actual[:, :3, 3] - a[:3, 3]) @ axis / (axis @ axis)
            geometric.append(
                {
                    "stage": labels[i],
                    "max_position_error_m": float(pe.max()),
                    "max_orientation_error_rad": float(oe.max()),
                    "max_backward_progress": float(
                        np.max(np.maximum.accumulate(progress) - progress)
                    ),
                }
            )
            samples.extend(np.column_stack((q, np.tile(fixture, (len(q), 1)))))
            piece_labels.extend([labels[i]] * len(q))
        samples = np.asarray(samples)
        stage_order = [before_stage["stage"], *labels, after_stage["stage"]]
        rank = {name: i for i, name in enumerate(stage_order)}
        ordering = sorted(range(len(samples)), key=lambda index: rank[piece_labels[index]])
        samples = samples[ordering]
        piece_labels = [piece_labels[index] for index in ordering]
        stage_position_gaps = [
            {
                "left": left,
                "right": right,
                "nine_axis_max": float(
                    np.max(
                        np.abs(
                            samples[piece_labels.index(right)]
                            - samples[len(piece_labels) - 1 - piece_labels[::-1].index(left)]
                        )
                    )
                ),
            }
            for left, right in zip(stage_order[:-1], stage_order[1:], strict=True)
        ]
        excess = np.max(
            np.maximum(np.maximum(limits[:, 0] - samples, samples - limits[:, 1]), 0), axis=0
        )
        hits = []
        for i, state in enumerate(samples):
            for hit in checker.contacts(state):
                if float(hit.distance_m) < 0:
                    hits.append(
                        {
                            "stage": piece_labels[i],
                            "sample": i,
                            "distance_m": float(hit.distance_m),
                            "links": [str(x) for x in hit.links],
                        }
                    )

        def gap(a, b):
            return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

        join_checks = [
            {
                "join": f"{before_stage['stage']} -> approach:0",
                "position_max": gap(joints[0]["end_position"], curves[0](0)),
                "velocity_max": gap(joints[0]["end_velocity"], velocity0),
                "acceleration_max": gap(joints[0]["end_acceleration"], accel0),
            },
            {
                "join": f"withdraw:0 -> {after_stage['stage']}",
                "position_max": gap(curves[-1](1), joints[1]["start_position"]),
                "velocity_max": gap(velocity1, joints[1]["start_velocity"]),
                "acceleration_max": gap(accel1, joints[1]["start_acceleration"]),
            },
        ]
        report = {
            "scope": "One fresh Tour09 probe and both adjacent generated transits, modeled clock and explicit matched spline derivatives, not the complete tour or controller timing; process geometry checked at finite samples, cubic process jerk can jump at internal knots. The unmodified next approach uses joint-linear secant and assumed zero acceleration at its beginning.",
            "generation_s": raw["generation_s"],
            "refine_transits_s": solve_s,
            "verification_s": perf_counter() - started - solve_s,
            "transits": results,
            "transit_join_diagnostics": joints,
            "probe_transit_joins": join_checks,
            "process_internal_joins": retime["continuity"],
            "process_geometry": geometric,
            "stage_sequence": stage_order,
            "nine_axis_stage_position_gaps": stage_position_gaps,
            "total_checked_samples": len(samples),
            "negative_contacts": len(hits),
            "worst_contact": min(hits, key=lambda x: x["distance_m"]) if hits else None,
            "raw_limit_excess": excess.tolist(),
            "source_generated_endpoint_normalization_max": normalization_max,
            "modeled_outer_start_velocity": start_velocity.tolist(),
            "modeled_next_approach_velocity": next_velocity.tolist(),
        }
        (output / "probe-neighbors-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            "PROBE_NEIGHBORS",
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in (
                        "transits",
                        "transit_join_diagnostics",
                        "modeled_outer_start_velocity",
                        "modeled_next_approach_velocity",
                    )
                }
            ),
            flush=True,
        )
    finally:
        _restore_scene(env, initial, revision)
    if snapshot(scene) != before:
        raise RuntimeError("scene changed")
