"""Generate tangent leads and local source-bead blends in one scene."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline  # noqa: F401  # Load selected native research libraries.
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401  # Verify selected SQP runtime.
from actual_stage_adapters import weld_stage
from actual_stage_contract import diagnostic_joins
from check_joint_motion_graph import LEAD_LENGTH_M
from cubic_position_extrema import cubic_position_extrema
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve
from generated_rail_reference import generated_rail_at
from master_analytic_ik import arm_ik
from probe_process_inputs import extract_process_tour
from probe_source_join_trial import pose_matrix
from process_geometry import assess_process_geometry
from rotation_targets import rotation_errors
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from weld_error_witness import select_error_witnesses
from weld_local_blend import build_local_weld_blend
from weld_only_inputs import WeldBead, extract_tour
from weld_source_orientation import build_source_bead_orientation

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent actual-curve collision sampling density.
SPEED_BOUND_ULPS = 32  # Binary64 allowance at the declared axis-speed bound.
NODE_TIME_ULPS = 16  # Distinguish duplicate representations of the same modeled time.
WITNESS_BUDGET_PER_BEAD = 3  # One bounded pass; at most one extra IK node per affected stage.
PROGRESS_ROUNDOFF_ULPS = 256  # Binary64 position-comparison allowance, not process tolerance.


class WeldLocalTrialError(RuntimeError):
    """A source bead, generated IK path or actual curve cannot be checked."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--lead-policy", choices=("straight-tangent", "matched-endpoints"), required=True
    )
    parser.add_argument(
        "--witness-refinement", action="store_true", help="add a bounded actual-FK witness pass"
    )
    parser.add_argument(
        "--generated-rail-reference",
        action="store_true",
        help="evaluate the generated rail reference at each process target",
    )
    parser.add_argument(
        "--source-bead-orientation",
        action="store_true",
        help="target original per-segment orientation during active bead stages",
    )
    parser.add_argument(
        "--orientation-witness-refinement",
        action="store_true",
        help="include actual orientation error in the bounded witness policy",
    )
    parser.add_argument(
        "--diagnostic-first-stage", help="evaluate one cluster identified by its first stage"
    )
    parser.add_argument(
        "--verified-recovery",
        action="store_true",
        help="reconstruct the modeled clock from an independently verified fresh recovery",
    )
    args = parser.parse_args()
    if args.orientation_witness_refinement and not args.witness_refinement:
        raise WeldLocalTrialError("orientation witnesses require the bounded witness pass")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    saved = json.loads(args.input.read_text())
    if args.verified_recovery:
        if saved["verified"]["status"] != "independently_sampled_tour":
            raise WeldLocalTrialError("independently verified fresh recovery required")
        candidate = None
    else:
        saved = saved["result"]
        candidate = saved["jerk"]
        if (
            saved["product"]["status"] != "independently_sampled_tour"
            or candidate["process_verification"]["status"] != "independently_sampled_tour"
        ):
            raise WeldLocalTrialError("independently verified fresh generated input required")
    if args.output.exists():
        raise WeldLocalTrialError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    original = snapshot(scene)
    env, program, source, tours, names, opw, limits7, _, _, home, _, _ = scene["inputs"]
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    if args.verified_recovery:
        tour = int(saved["verified"]["tour"])
        raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
        segments, clock, _ = prepare_fresh(
            env,
            program,
            source,
            tours,
            home,
            names,
            raw,
            metadata,
            tour,
            saved["record"],
            saved["verified"],
        )
        candidate = {"tour": tour, "generated_stages": segments, "clock": clock}
    speeds = []
    for name in names[:7]:
        item = metadata[name]
        if item["speed_source"] not in ("datasheet", "cell_declared"):
            raise WeldLocalTrialError(f"missing sourced speed for {name}")
        speeds.append(float(item["velocity"]))
    speeds = np.asarray(speeds)
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("local_weld_batch_nine", list(names)).getLimits().joint_limits
    )
    task = extract_tour(source, tours, int(candidate["tour"]), "workpiece_link", home=home)
    process = extract_process_tour(program, task, probe_metadata_frame="workpiece_link")
    beads = [item for item in process.operations if isinstance(item, WeldBead)]
    rows = candidate["generated_stages"]
    clocks = {entry["stage"]: entry for entry in candidate["clock"]["stages"]}
    runs = []
    for row in rows:
        if row["stage"].startswith("weld:"):
            if not runs or runs[-1] is None:
                runs.append([])
            runs[-1].append(row)
        elif runs and runs[-1] is not None:
            runs.append(None)
    runs = [run for run in runs if run is not None]
    if len(runs) != len(beads):
        raise WeldLocalTrialError("generated weld clusters do not match source beads")
    if args.diagnostic_first_stage and not any(
        run[0]["stage"] == args.diagnostic_first_stage for run in runs
    ):
        raise WeldLocalTrialError("diagnostic stage is not a weld-cluster start")
    report = {
        "tour": candidate["tour"],
        "scene_setup_s_contended": setup,
        "beads": [],
        "lead_policy": args.lead_policy,
        "witness_refinement": args.witness_refinement,
        "orientation_policy": "source bead targets"
        if args.source_bead_orientation
        else "local blends",
        "orientation_witness_refinement": args.orientation_witness_refinement,
        "rail_policy": "generated reference, not optimized"
        if args.generated_rail_reference
        else "fixed",
        "scope": "Exploratory generated tangent leads and local source-bead Cartesian blends through source waypoints; finite samples and modeled clock, no process or controller acceptance.",
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

    for bead, run in zip(beads, runs, strict=True):
        if args.diagnostic_first_stage and run[0]["stage"] != args.diagnostic_first_stage:
            continue
        result = {"source_segment": bead.segment, "status": "in_progress", "stages": []}
        try:
            source_poses = np.asarray([pose_matrix(item) for item in bead.poses])
            if len(source_poses) < 2 or len(run) != len(source_poses) + 1:
                raise WeldLocalTrialError("source bead and generated stages differ")
            fixture = np.asarray(run[0]["states"][0][7:], dtype=float)
            all_source = [np.asarray(row["states"], dtype=float) for row in run]
            if any(not np.all(q[:, 7:] == fixture) for q in all_source):
                raise WeldLocalTrialError("moving fixture during weld is unsupported")
            track = float(all_source[0][0, 0])
            if not args.generated_rail_reference and any(
                not np.all(q[:, 0] == track) for q in all_source
            ):
                raise WeldLocalTrialError(
                    "generated moving-track bead requires a different IK hypothesis"
                )
            bead_points = source_poses[:, :3, 3]
            delta_first = bead_points[1] - bead_points[0]
            delta_last = bead_points[-1] - bead_points[-2]
            lengths = (float(np.linalg.norm(delta_first)), float(np.linalg.norm(delta_last)))
            if min(lengths) == 0:
                raise WeldLocalTrialError("zero-length bead endpoint segment")
            if args.lead_policy == "straight-tangent":
                lead_first = source_poses[0].copy()
                lead_last = source_poses[-1].copy()
                lead_first[:3, 3] = bead_points[0] - float(LEAD_LENGTH_M) * delta_first / lengths[0]
                lead_last[:3, 3] = bead_points[-1] + float(LEAD_LENGTH_M) * delta_last / lengths[1]
            else:
                lead_first = pose(all_source[0][0])
                lead_last = pose(all_source[-1][-1])
            points = np.vstack((lead_first[:3, 3], bead_points, lead_last[:3, 3]))
            rotations = np.concatenate(
                (lead_first[None, :3, :3], source_poses[:, :3, :3], lead_last[None, :3, :3]), axis=0
            )
            durations = np.asarray([float(clocks[row["stage"]]["duration_s"]) for row in run])
            position, orientation, blend = build_local_weld_blend(points, rotations, durations)
            if args.source_bead_orientation:
                orientation = build_source_bead_orientation(rotations, durations, orientation)
            edges = np.r_[0.0, np.cumsum(durations)]
            fixture_map = dict(zip(names[7:], map(float, fixture), strict=True))
            carrier = {**fixture_map, names[0]: track}
            part = np.asarray(env.getState(fixture_map).link_transforms["workpiece_link"].matrix)
            qref = all_source[0][0, :7].copy()
            time_nodes = []
            joint_nodes = []
            counts = []
            clear_counts = []
            began = perf_counter()
            for j, row in enumerate(run):
                local = np.linspace(0.0, durations[j], len(all_source[j]))
                for start, end in blend["position_windows_s"][j]:
                    local = np.r_[local, start, end]
                local = np.unique(local)
                keep = np.r_[
                    True,
                    np.diff(local)
                    > NODE_TIME_ULPS
                    * np.finfo(float).eps
                    * np.maximum(1.0, np.maximum(np.abs(local[:-1]), np.abs(local[1:]))),
                ]
                local = local[keep]
                local[0], local[-1] = 0.0, durations[j]
                for t in local:
                    global_time = float(edges[j] + t)
                    if time_nodes and global_time == time_nodes[-1]:
                        continue
                    target = np.eye(4)
                    target[:3, 3] = position(j, float(t))
                    target[:3, :3] = orientation(j, float(t))
                    if args.generated_rail_reference:
                        track = float(generated_rail_at(all_source[j], durations[j], float(t)))
                        carrier[names[0]] = track
                    solution = arm_ik(
                        env,
                        opw,
                        part @ target,
                        carrier,
                        tcp_link="nozzle_link",
                        limits=np.asarray(limits7)[1:],
                    )
                    counts.append(len(solution.solutions))
                    if not solution.solutions:
                        raise WeldLocalTrialError(
                            f"{row['stage']} at {t} s has no generated IK branch"
                        )
                    choices = [
                        np.r_[track, arm]
                        for arm in solution.solutions
                        if not any(
                            float(hit.distance_m) < 0
                            for hit in checker.contacts(np.r_[track, arm, fixture])
                        )
                    ]
                    clear_counts.append(len(choices))
                    if not choices:
                        raise WeldLocalTrialError(
                            f"{row['stage']} at {t} s has no clear generated IK branch"
                        )
                    qref = min(choices, key=lambda q: float(np.sum(((q - qref) / speeds) ** 2)))
                    time_nodes.append(global_time)
                    joint_nodes.append(qref)
            result["ik_s_contended"] = perf_counter() - began
            time_nodes = np.asarray(time_nodes)
            joint_nodes = np.asarray(joint_nodes)
            if np.any(
                np.abs(np.diff(joint_nodes, axis=0)) / np.diff(time_nodes)[:, None] / speeds
                > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps
            ):
                raise WeldLocalTrialError("generated IK branch step exceeds sourced axis speed")
            curve = CubicSpline(time_nodes, joint_nodes, axis=0)
            result["initial_node_count"] = len(time_nodes)
            result["witnesses"] = []
            if args.witness_refinement:
                began = perf_counter()
                witness_candidates = []
                error_stages = []
                local_times = []
                for j, row in enumerate(run):
                    local = np.linspace(0.0, durations[j], SAMPLES)
                    times = edges[j] + local
                    samples = curve(times)
                    actual = np.asarray([pose(np.r_[q, fixture]) for q in samples])
                    target = np.asarray([position(j, float(t)) for t in local])
                    error = np.linalg.norm(actual[:, :3, 3] - target, axis=1)
                    if args.orientation_witness_refinement:
                        desired = np.asarray([orientation(j, float(t)) for t in local])
                        angle = rotation_errors(desired, actual[:, :3, :3])
                        eligible = np.asarray(
                            [
                                not np.any(
                                    np.abs(time_nodes - t)
                                    <= NODE_TIME_ULPS * np.finfo(float).eps * max(1.0, abs(t))
                                )
                                for t in times
                            ]
                        )
                        error_stages.append((error, angle, eligible))
                        local_times.append(local)
                        continue
                    order = np.argsort(error)[::-1]
                    for index in order:
                        t = float(times[index])
                        band = NODE_TIME_ULPS * np.finfo(float).eps * max(1.0, abs(t))
                        if not np.any(np.abs(time_nodes - t) <= band):
                            witness_candidates.append(
                                (float(error[index]), j, float(local[index]), t)
                            )
                            break
                if args.orientation_witness_refinement:
                    for j, index in select_error_witnesses(error_stages, WITNESS_BUDGET_PER_BEAD):
                        local_t = float(local_times[j][index])
                        error = float(error_stages[j][0][index])
                        witness_candidates.append((error, j, local_t, float(edges[j] + local_t)))
                witness_candidates.sort(reverse=True)
                for error, j, local_t, global_t in witness_candidates[:WITNESS_BUDGET_PER_BEAD]:
                    target = np.eye(4)
                    target[:3, 3] = position(j, local_t)
                    target[:3, :3] = orientation(j, local_t)
                    if args.generated_rail_reference:
                        track = float(generated_rail_at(all_source[j], durations[j], local_t))
                        carrier[names[0]] = track
                    solution = arm_ik(
                        env,
                        opw,
                        part @ target,
                        carrier,
                        tcp_link="nozzle_link",
                        limits=np.asarray(limits7)[1:],
                    )
                    choices = []
                    for arm in solution.solutions:
                        point = np.r_[track, arm]
                        if not any(
                            float(hit.distance_m) < 0
                            for hit in checker.contacts(np.r_[point, fixture])
                        ):
                            choices.append(point)
                    if not choices:
                        raise WeldLocalTrialError(
                            f"{run[j]['stage']}: no clear IK branch at selected target witness"
                        )
                    index = int(np.searchsorted(time_nodes, global_t))
                    if index == 0 or index == len(time_nodes):
                        raise WeldLocalTrialError(
                            "interior witness must have two neighboring nodes"
                        )
                    left, right = joint_nodes[index - 1], joint_nodes[index]
                    chosen = min(
                        choices,
                        key=lambda q: float(
                            np.sum(((q - left) / speeds) ** 2) + np.sum(((q - right) / speeds) ** 2)
                        ),
                    )
                    time_nodes = np.insert(time_nodes, index, global_t)
                    joint_nodes = np.insert(joint_nodes, index, chosen, axis=0)
                    result["witnesses"].append(
                        {
                            "stage": run[j]["stage"],
                            "modeled_time_s": global_t,
                            "initial_target_error_m": error,
                            "initial_target_orientation_error_rad": (
                                float(error_stages[j][1][np.searchsorted(local_times[j], local_t)])
                                if args.orientation_witness_refinement
                                else None
                            ),
                            "clear_ik_branches": len(choices),
                        }
                    )
                if np.any(
                    np.abs(np.diff(joint_nodes, axis=0)) / np.diff(time_nodes)[:, None] / speeds
                    > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps
                ):
                    raise WeldLocalTrialError(
                        "refined generated IK steps exceed sourced axis speed"
                    )
                curve = CubicSpline(time_nodes, joint_nodes, axis=0)
                result["witness_refinement_s_contended"] = perf_counter() - began
            result["final_node_count"] = len(time_nodes)
            extrema = cubic_position_extrema(curve)
            excess = [
                max(limits[i, 0] - item["minimum"], item["maximum"] - limits[i, 1], 0.0)
                for i, item in enumerate(extrema)
            ]
            if any(value > SQP_FEASIBILITY_BAND for value in excess):
                raise CurveLimitError("actual cubic position exceeds configured band")
            stages = [
                weld_stage(
                    row["stage"], float(durations[j]), curve, float(edges[j]), tuple(fixture)
                )
                for j, row in enumerate(run)
            ]
            result["modeled_joins"] = [join._asdict() for join in diagnostic_joins(stages)]
            result["source_waypoint_errors"] = []
            for j, t in enumerate(edges[1:-1], start=1):
                actual = pose(np.r_[curve(t), fixture])
                result["source_waypoint_errors"].append(
                    {
                        "source_index": j - 1,
                        "position_m": float(np.linalg.norm(actual[:3, 3] - points[j])),
                        "orientation_rad": float(
                            rotation_errors(rotations[j : j + 1], actual[None, :3, :3])[0]
                        ),
                    }
                )
            failed = False
            unresolved_order = []
            for j, stage in enumerate(stages):
                local = np.linspace(0.0, stage.duration, SAMPLES)
                times = edges[j] + local
                q7 = curve(times)
                vel = curve(times, 1)
                q = np.column_stack((q7, np.tile(fixture, (len(q7), 1))))
                actual = np.asarray([pose(point) for point in q])
                first = np.eye(4)
                first[:3, 3] = points[j]
                first[:3, :3] = rotations[j]
                last = np.eye(4)
                last[:3, 3] = points[j + 1]
                last[:3, :3] = rotations[j + 1]
                geometry = assess_process_geometry(first, last, actual)
                desired_position = np.asarray([position(j, float(t)) for t in local])
                desired_orientation = np.asarray([orientation(j, float(t)) for t in local])
                desired_pose = np.tile(np.eye(4), (len(local), 1, 1))
                desired_pose[:, :3, 3] = desired_position
                desired_pose[:, :3, :3] = desired_orientation
                target_geometry = assess_process_geometry(first, last, desired_pose)
                progress_band = (
                    PROGRESS_ROUNDOFF_ULPS
                    * np.finfo(float).eps
                    * max(
                        1.0, float(np.linalg.norm(first[:3, 3])), float(np.linalg.norm(last[:3, 3]))
                    )
                )
                if j not in (0, len(stages) - 1) and (
                    geometry["max_backward_distance_m"] > progress_band
                    or target_geometry["max_backward_distance_m"] > progress_band
                ):
                    unresolved_order.append(stage.label)
                target_error = float(
                    np.linalg.norm(actual[:, :3, 3] - desired_position, axis=1).max()
                )
                orientation_error = float(
                    rotation_errors(desired_orientation, actual[:, :3, :3]).max()
                )
                try:
                    collision = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
                except (CurveCollisionError, CurveLimitError) as exc:
                    collision = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failed = True
                ratios = np.max(np.abs(vel), axis=0) / speeds
                speed_pass = bool(np.all(ratios <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps))
                if not speed_pass:
                    failed = True
                feed = np.linalg.norm(np.gradient(actual[:, :3, 3], local, axis=0), axis=1)
                nominal = float(np.linalg.norm(points[j + 1] - points[j]) / durations[j])
                jerk = np.asarray(curve(times, 3))
                result["stages"].append(
                    {
                        "stage": stage.label,
                        "kind": "generated_lead" if j in (0, len(stages) - 1) else "source_bead",
                        "duration_s": float(stage.duration),
                        "geometry": geometry,
                        "target_geometry": target_geometry,
                        "blend_target_position_error_m": target_error,
                        "blend_target_orientation_error_rad": orientation_error,
                        "nominal_feed_m_s": nominal,
                        "modeled_feed_range_m_s": [float(feed.min()), float(feed.max())],
                        "sampled_speed_ratio": ratios.tolist(),
                        "sampled_speeds_pass": speed_pass,
                        "collision_and_position": collision,
                        "sampled_peak_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                    }
                )
                save()
            status = (
                "failed_sampled_check"
                if failed
                else (
                    "sampled_bead_order_unresolved"
                    if unresolved_order
                    else "sampled_exploratory_weld_only"
                )
            )
            result.update(
                {
                    "status": status,
                    "unresolved_bead_order_stages": unresolved_order,
                    "blend": blend,
                    "minimum_ik_branch_count": min(counts),
                    "minimum_clear_ik_branch_count": min(clear_counts),
                    "ik_queries": len(counts) + len(result["witnesses"]),
                    "raw_position_extrema_excess": excess,
                    "times_s": time_nodes.tolist(),
                    "joint_states_7": joint_nodes.tolist(),
                    "fixture": fixture.tolist(),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        report["beads"].append(result)
        save()
        print(
            json.dumps(
                {
                    "source_segment": bead.segment,
                    "status": result["status"],
                    "stages": len(result["stages"]),
                    "error": result.get("error"),
                }
            ),
            flush=True,
        )
    if snapshot(scene) != original:
        raise WeldLocalTrialError("scene state changed")


if __name__ == "__main__":
    main()
