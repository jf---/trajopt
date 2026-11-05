"""Isolated generated probe research experiment."""

import json
from time import perf_counter

import numpy as np
from free_space_transit import _restore_scene
from fresh_scene import snapshot
from master_analytic_ik import arm_ik
from rotation_targets import interpolate_rotations, rotation_errors

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters


def run_ik_progress(scene, segments, clock, speed, output):
    before = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]

    labels = ["approach:0", "search:0", "retract:0", "withdraw:0"]
    stages = [next(s for s in segments if s["stage"] == label) for label in labels]
    first = next(i for i, s in enumerate(segments) if s["stage"] == labels[0])
    durations = [float(clock["stages"][i]["duration_s"]) for i in range(first, first + 4)]
    edges = np.r_[0.0, np.cumsum(durations)]
    fixture = np.asarray(stages[0]["states"][0][7:])
    track = float(stages[0]["states"][0][0])
    held = dict(zip(names[7:], map(float, fixture), strict=True))
    carrier = {**held, names[0]: track}
    initial = {str(k): float(v) for k, v in env.getState().joints.items()}
    revision = int(env.getRevision())
    begin = perf_counter()
    try:
        env.setState(held)
        group = env.getJointGroup(info.manipulator)
        limits = np.asarray(group.getLimits().joint_limits)
        opw = env.getKinematicGroup("fresh_pool_opw")

        def pose(q):
            state = env.getState({**held, **dict(zip(names[:7], map(float, q), strict=True))})
            return np.linalg.solve(
                np.asarray(state.link_transforms["workpiece_link"].matrix),
                np.asarray(state.link_transforms["nozzle_link"].matrix),
            )

        part = np.asarray(env.getState(held).link_transforms["workpiece_link"].matrix)
        targets = []
        samples = 17
        for i, stage in enumerate(stages):
            a = pose(np.asarray(stage["states"][0])[:7])
            b = pose(np.asarray(stage["states"][-1])[:7])
            sample_progress = np.linspace(0, 1, samples)
            rotations = interpolate_rotations(a[:3, :3], b[:3, :3], sample_progress)
            matrices = []
            for u, rotation in zip(sample_progress, rotations, strict=True):
                target = np.eye(4)
                target[:3, 3] = (1 - u) * a[:3, 3] + u * b[:3, 3]
                target[:3, :3] = rotation
                matrices.append(target)
            targets.append(matrices)
        previous = np.asarray(stages[0]["states"][0], dtype=float)[:7]
        leg_states = []
        counts = []
        solving_start = perf_counter()
        for i in range(len(stages)):
            states = []
            for j, target in enumerate(targets[i]):
                if i > 0 and j == 0:
                    states.append(previous.copy())
                    continue
                solved = arm_ik(
                    env, opw, part @ target, carrier, tcp_link="nozzle_link", limits=limits[1:]
                )
                candidates = [np.r_[track, q] for q in solved.solutions]
                counts.append(len(candidates))
                if not candidates:
                    raise RuntimeError(f"no generated IK branch for leg {i} sample {j}")
                q = min(candidates, key=lambda q: float(np.sum(((q - previous) / speed) ** 2)))
                states.append(q)
                previous = q
            leg_states.append(np.asarray(states))
        solve_s = perf_counter() - solving_start
        verify_start = perf_counter()
        checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
        legs = []
        all_hits = []
        all_excess = np.zeros(9)
        dvel = []
        limits9 = np.asarray(
            env.getJointGroup("ik-progress-nine", list(names)).getLimits().joint_limits
        )
        for i, (states, wanted) in enumerate(zip(leg_states, targets, strict=True)):
            q = np.asarray(states)
            times = np.linspace(edges[i], edges[i + 1], len(q))
            mid = 0.5 * (q[:-1] + q[1:])
            queries = np.empty((2 * len(q) - 1, 7))
            queries[0::2] = q
            queries[1::2] = mid
            progress = np.linspace(0, 1, len(queries))
            a = wanted[0]
            b = wanted[-1]
            rots = interpolate_rotations(a[:3, :3], b[:3, :3], progress)
            got = np.asarray([pose(x) for x in queries])
            desired = a[:3, 3] + progress[:, None] * (b[:3, 3] - a[:3, 3])
            perr = np.linalg.norm(got[:, :3, 3] - desired, axis=1)
            angle = rotation_errors(rots, got[:, :3, :3])
            axis = b[:3, 3] - a[:3, 3]
            actual_progress = (got[:, :3, 3] - a[:3, 3]) @ axis / (axis @ axis)
            full = np.column_stack((queries, np.tile(fixture, (len(queries), 1))))
            all_excess = np.maximum(
                all_excess,
                np.max(
                    np.maximum(np.maximum(limits9[:, 0] - full, full - limits9[:, 1]), 0), axis=0
                ),
            )
            hits = []
            for j, state in enumerate(full):
                for hit in checker.contacts(state):
                    if float(hit.distance_m) < 0:
                        hits.append(
                            {
                                "stage": labels[i],
                                "sample_index": j,
                                "distance_m": float(hit.distance_m),
                                "links": [str(x) for x in hit.links],
                            }
                        )
            all_hits.extend(hits)
            vel = np.diff(q, axis=0) / np.diff(times)[:, None]
            dvel.append(vel)
            legs.append(
                {
                    "stage": labels[i],
                    "nodes": len(q),
                    "ik_branch_counts": counts[(i * (samples - 1)) : (i + 1) * (samples - 1)],
                    "node_max_fk_error_m": float(perr[0::2].max()),
                    "midpoint_max_fk_error_m": float(perr[1::2].max()),
                    "node_max_orientation_error_rad": float(angle[0::2].max()),
                    "midpoint_max_orientation_error_rad": float(angle[1::2].max()),
                    "progress_min": float(actual_progress.min()),
                    "progress_max": float(actual_progress.max()),
                    "maximum_backward_progress": float(
                        np.max(np.maximum.accumulate(actual_progress) - actual_progress)
                    ),
                    "negative_contacts": len(hits),
                    "maximum_sample_joint_step_rad": float(
                        np.max(np.abs(np.diff(q[:, 1:], axis=0)))
                    ),
                }
            )
        joins = []
        for i in range(3):
            v_left = dvel[i][-1]
            v_right = dvel[i + 1][0]
            joins.append(
                {
                    "left": labels[i],
                    "right": labels[i + 1],
                    "max_sampled_secant_jump_rad_s": float(
                        np.max(np.abs(v_right[1:] - v_left[1:]))
                    ),
                    "left_secant": v_left.tolist(),
                    "right_secant": v_right.tolist(),
                    "joint_endpoint_gap_max": float(
                        np.max(np.abs(leg_states[i][-1] - leg_states[i + 1][0]))
                    ),
                }
            )
        report = {
            "scope": "Fresh generated Tour09 probe, OPW IK continuation at fixed generated track, 17 samples per process leg, piecewise joint-linear between IK samples and source-derived target pose interpolation. Modeled clock only; no solver jerk optimization, actual controller timing, unobserved hit or complete-tour claim.",
            "generated_ik_calls": len(counts),
            "generation_s": solve_s,
            "verify_s": perf_counter() - verify_start,
            "total_s_excluding_scene": perf_counter() - begin,
            "ik_branch_count_min": min(counts),
            "ik_branch_count_max": max(counts),
            "legs": legs,
            "joins": joins,
            "negative_contacts": len(all_hits),
            "worst_contact": min(all_hits, key=lambda x: x["distance_m"]) if all_hits else None,
            "raw_limit_excess": all_excess.tolist(),
            "first_generated_start_gap_rad": float(
                np.max(np.abs(leg_states[0][0] - np.asarray(stages[0]["states"][0])[:7]))
            ),
            "last_generated_end_gap_rad": float(
                np.max(np.abs(leg_states[-1][-1] - np.asarray(stages[-1]["states"][-1])[:7]))
            ),
            "fixture": fixture.tolist(),
            "track_m": track,
            "stage_states": [q.tolist() for q in leg_states],
            "edges_s": edges.tolist(),
        }
        (output / "ik-progress-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            "IK_PROGRESS_RESULT",
            json.dumps({k: v for k, v in report.items() if k != "stage_states"}),
            flush=True,
        )
    finally:
        _restore_scene(env, initial, revision)
    if snapshot(scene) != before:
        raise RuntimeError("scene changed")
