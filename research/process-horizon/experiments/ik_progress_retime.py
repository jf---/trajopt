"""Isolated generated probe research experiment."""

import json
from time import perf_counter

import numpy as np
from free_space_transit import _restore_scene
from fresh_scene import snapshot
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters


def run_ik_progress_retime(scene, segments, clock, speed, output):
    before = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]

    d = json.loads((output / "ik-progress-report.json").read_text())
    labels = ["approach:0", "search:0", "retract:0", "withdraw:0"]
    stages = [next(s for s in segments if s["stage"] == label) for label in labels]
    edges = np.asarray(d["edges_s"])
    fixture = np.asarray(d["fixture"])
    states = [np.asarray(x) for x in d["stage_states"]]
    splines = [CubicSpline(np.linspace(0, 1, len(q)), q, axis=0) for q in states]
    initial = {str(k): float(v) for k, v in env.getState().joints.items()}
    revision = int(env.getRevision())
    start = perf_counter()
    try:
        env.setState(dict(zip(names[7:], map(float, fixture), strict=True)))
        group = env.getJointGroup(info.manipulator)

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

        joins = []
        for i in range(3):
            q = states[i][-1]
            v0 = splines[i](1.0, 1)
            v1 = splines[i + 1](0.0, 1)
            j = np.asarray(group.calcJacobian(q, "nozzle_link"))
            tw0 = j @ v0
            tw1 = j @ v1
            matrix = np.column_stack((tw0, tw1))
            sigma = np.linalg.svd(matrix, compute_uv=False)
            rank = np.linalg.matrix_rank(matrix)
            alignment = float(tw0 @ tw1 / (np.linalg.norm(tw0) * np.linalg.norm(tw1)))
            # Non-collinear or opposing process twists cannot pass through at nonzero TCP speed.
            if rank < 2 and alignment > 0:
                raise RuntimeError(
                    "compatible join needs shared nonzero velocity law; no blanket stop allowed"
                )
            joins.append(
                {
                    "left": labels[i],
                    "right": labels[i + 1],
                    "tcp_twist_rank": int(rank),
                    "tcp_twist_singular_values": sigma.tolist(),
                    "twist_cosine": alignment,
                    "reason": "noncollinear or opposing modeled process tangents; zero TCP velocity required",
                }
            )
        checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
        limits = np.asarray(
            env.getJointGroup("ik-retime-nine", list(names)).getLimits().joint_limits
        )
        legs = []
        all_hits = []
        excess = np.zeros(9)
        vel_by_leg = []
        acc_by_leg = []
        jerk_by_leg = []
        for i, (stage, curve) in enumerate(zip(stages, splines, strict=True)):
            # Preserve nonzero outer progress rates; true process corners stop only internally.
            v0 = 1.0 if i == 0 else 0.0
            v1 = 1.0 if i == 3 else 0.0
            tau = np.linspace(0, 1, 257)
            duration = edges[i + 1] - edges[i]
            a3 = 10 - 6 * v0 - 4 * v1
            a4 = -15 + 8 * v0 + 7 * v1
            a5 = 6 - 3 * v0 - 3 * v1
            s = v0 * tau + a3 * tau**3 + a4 * tau**4 + a5 * tau**5
            sd = (v0 + 3 * a3 * tau**2 + 4 * a4 * tau**3 + 5 * a5 * tau**4) / duration
            sdd = (6 * a3 * tau + 12 * a4 * tau**2 + 20 * a5 * tau**3) / duration**2
            sddd = (6 * a3 + 24 * a4 * tau + 60 * a5 * tau**2) / duration**3
            if np.min(sd) < -np.finfo(float).eps * 128:
                raise RuntimeError("modeled scalar clock reverses")
            s = np.clip(s, 0, 1)
            q = curve(s)
            qs = curve(s, 1)
            qss = curve(s, 2)
            qsss = curve(s, 3)
            v = qs * sd[:, None]
            acc = qss * sd[:, None] ** 2 + qs * sdd[:, None]
            jerk = (
                qsss * sd[:, None] ** 3 + 3 * qss * sd[:, None] * sdd[:, None] + qs * sddd[:, None]
            )
            vel_by_leg.append(v)
            acc_by_leg.append(acc)
            jerk_by_leg.append(jerk)
            a = pose(np.asarray(stage["states"][0])[:7])
            b = pose(np.asarray(stage["states"][-1])[:7])
            actual = np.asarray([pose(x) for x in q])
            p = actual[:, :3, 3]
            target = a[:3, 3] + s[:, None] * (b[:3, 3] - a[:3, 3])
            desired = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3]))))(s)
            perr = np.linalg.norm(p - target, axis=1)
            angle = (desired.inv() * Rotation.from_matrix(actual[:, :3, :3])).magnitude()
            full = np.column_stack((q, np.tile(fixture, (len(q), 1))))
            excess = np.maximum(
                excess,
                np.max(np.maximum(np.maximum(limits[:, 0] - full, full - limits[:, 1]), 0), axis=0),
            )
            hits = []
            for j, state in enumerate(full):
                for h in checker.contacts(state):
                    if float(h.distance_m) < 0:
                        hits.append(
                            {
                                "stage": labels[i],
                                "sample": j,
                                "distance_m": float(h.distance_m),
                                "links": [str(x) for x in h.links],
                            }
                        )
            all_hits.extend(hits)
            legs.append(
                {
                    "stage": labels[i],
                    "samples": len(q),
                    "duration_s": float(duration),
                    "max_tcp_position_error_m": float(perr.max()),
                    "max_tcp_orientation_error_rad": float(angle.max()),
                    "progress_min": float(s.min()),
                    "progress_max": float(s.max()),
                    "min_progress_rate": float(sd.min()),
                    "max_speed_ratio": float(np.max(np.abs(v) / np.asarray(speed))),
                    "max_abs_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                    "negative_contacts": len(hits),
                }
            )
        continuity = []
        for i in range(3):
            continuity.append(
                {
                    "left": labels[i],
                    "right": labels[i + 1],
                    "position_gap_max": float(np.max(np.abs(states[i][-1] - states[i + 1][0]))),
                    "velocity_gap_max": float(
                        np.max(np.abs(vel_by_leg[i][-1] - vel_by_leg[i + 1][0]))
                    ),
                    "acceleration_gap_max": float(
                        np.max(np.abs(acc_by_leg[i][-1] - acc_by_leg[i + 1][0]))
                    ),
                }
            )
        result = {
            "scope": "Fresh generated Tour09 probe only. Per-leg cubic joint splines interpolate OPW states; modeled quintic scalar timing stops only at three detected incompatible process-twist joins. Original modeled durations retained; outer process velocities nonzero. No jerk bound, actual controller dynamics or whole-tour claim. Cubic geometry is C2 per leg; jerk can jump at its internal knots.",
            "time_s_excluding_scene": perf_counter() - start,
            "joins": joins,
            "legs": legs,
            "continuity": continuity,
            "negative_contacts": len(all_hits),
            "worst_contact": min(all_hits, key=lambda x: x["distance_m"]) if all_hits else None,
            "raw_limit_excess": excess.tolist(),
            "first_process_velocity": vel_by_leg[0][0].tolist(),
            "last_process_velocity": vel_by_leg[-1][-1].tolist(),
            "outer_accelerations": [acc_by_leg[0][0].tolist(), acc_by_leg[-1][-1].tolist()],
        }
        (output / "ik-progress-retime.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print("IK_RETIME_RESULT", json.dumps(result), flush=True)
    finally:
        _restore_scene(env, initial, revision)
    if snapshot(scene) != before:
        raise RuntimeError("scene changed")
