"""Independently check all actual curves of a generated multi-cluster tour."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import CURVE_PARITY_ULPS, transit_stage, weld_stage
from actual_stage_contract import ActualStage, Seconds, StationaryEvent, diagnostic_joins
from actual_tour_assembly import assemble_actual_tour
from continuous_spline_jerk import build_jerk_model
from cubic_position_extrema import cubic_position_extrema
from fixture_easing_policy import ease_fixture_chains
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveLimitError, inspect_sampled_curve
from generated_knot_coalescing import SHORT_INTERVAL_FRACTION
from probe_cruise_clock import search_cruise_clock
from probe_cruise_trial import one_sided_progress
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import interpolate_rotations, rotation_errors
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from stage_local_weld_adapter import optimization_window, stage_local_curves
from timed_weld_tour import preserve_scene
from weld_local_blend import build_local_weld_blend
from weld_piecewise_correction import correct_piecewise_weld
from weld_tangent_leads import fit_lead

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

TIMED_POSE_WEIGHT = 1.0  # Same dimensionless objective as the Tour20 trial.
PROCESS_WEIGHT = 10.0  # Shared three-block process objective.
INITIAL_TRUST_BOX = 0.01  # Shared mixed-axis SQP trust-box policy.
SAMPLES = 1025  # Existing independent actual-curve collision sampling density.
MACHINE_ROUNDOFF_ULPS = 256  # Numerical comparison only, not a process tolerance.


class MultiProbeVerificationError(RuntimeError):
    """The generated tour or an independently evaluated curve is invalid."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "batch",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--exploratory-welds",
        action="store_true",
        help="verify explicit exploratory weld curves in the batch",
    )
    parser.add_argument(
        "--timed-pose-weld-batch",
        type=Path,
        help="opt in to stage-local refinement using the original generated blend evidence",
    )
    parser.add_argument("--coalescing-fraction", type=float, default=SHORT_INTERVAL_FRACTION)
    parser.add_argument("--coalescing-preview", action="store_true")
    parser.add_argument("--timed-pose-weights", type=float, nargs="+", default=[TIMED_POSE_WEIGHT])
    parser.add_argument("--timed-pose-modes", choices=["full", "bead"], nargs="+", default=["full"])
    args = parser.parse_args()
    if any(not np.isfinite(weight) or weight <= 0 for weight in args.timed_pose_weights):
        raise MultiProbeVerificationError("timed pose weights must be positive and finite")
    if len(set(args.timed_pose_weights)) != len(args.timed_pose_weights) or len(
        set(args.timed_pose_modes)
    ) != len(args.timed_pose_modes):
        raise MultiProbeVerificationError("timed pose grid settings must be distinct")
    if (
        args.timed_pose_weights != [TIMED_POSE_WEIGHT] or args.timed_pose_modes != ["full"]
    ) and not args.timed_pose_weld_batch:
        raise MultiProbeVerificationError("timed pose grid requires an explicit weld batch")
    grid_mode = len(args.timed_pose_weights) * len(args.timed_pose_modes) > 1
    if args.coalescing_preview and not args.timed_pose_weld_batch:
        raise MultiProbeVerificationError("coalescing preview requires timed pose welds")
    if args.timed_pose_weld_batch and not args.exploratory_welds:
        raise MultiProbeVerificationError("timed pose welds require explicit exploratory welds")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    saved = json.loads(args.input.read_text())["result"]
    candidate = saved["jerk"]
    batch = json.loads(args.batch.read_text())
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
        or batch["status"] != "optimization_completed_awaiting_actual_curve_verification"
    ):
        raise MultiProbeVerificationError(
            "fresh verified source and completed generated batch required"
        )
    if args.output.exists():
        raise MultiProbeVerificationError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    original = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    limits = np.asarray(
        env.getJointGroup("multi_probe_verify_nine", list(names)).getLimits().joint_limits
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise MultiProbeVerificationError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    speeds = np.asarray(speeds)

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    rows = candidate["generated_stages"]
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "stages": [],
        "scope": "Independent finite samples of the generated actual tour; fixture clocks modeled, actual hit and controller timing unknown.",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        clock, fixture_stages = ease_fixture_chains(rows, batch["clock"])
        replacements = dict(fixture_stages)
        curves = {}
        curve_offsets = {}
        weld_labels = set()
        source_bead_labels = set()
        weld_targets = {}
        phase_info = {}
        expected = [
            row["stage"].split(":")[1] for row in rows if row["stage"].startswith("approach:")
        ]
        if [group["probe"] for group in batch["groups"]] != expected:
            raise MultiProbeVerificationError(
                "generated batch does not cover source probes in order"
            )
        for group in batch["groups"]:
            for entry in group["process"]:
                source = next(row for row in rows if row["stage"] == entry["stage"])
                duration = float(entry["duration_s"])
                nodes = np.asarray(entry["ik_states"], dtype=float)
                times = np.linspace(0, duration, len(nodes))
                curve = fit_lead(
                    times,
                    nodes,
                    (np.asarray(entry["v0"]), np.asarray(entry["a0"])),
                    (np.asarray(entry["v1"]), np.asarray(entry["a1"])),
                )
                fixture = tuple(np.asarray(source["states"], dtype=float)[0, 7:])
                replacements[entry["stage"]] = weld_stage(
                    entry["stage"], duration, curve, 0.0, fixture
                )
                curves[entry["stage"]] = curve
                phase_info[entry["stage"]] = entry
        if args.exploratory_welds:
            if not batch.get("welds"):
                raise MultiProbeVerificationError("explicit exploratory weld data required")
            source_by_label = {row["stage"]: row for row in rows}
            timing_by_label = {row["stage"]: row for row in clock["stages"]}
            source_beads = {}
            if args.timed_pose_weld_batch:
                original_welds = json.loads(args.timed_pose_weld_batch.read_text())
                source_beads = {item["source_segment"]: item for item in original_welds["beads"]}
                expected_segments = {item["source_segment"] for item in batch["welds"]}
                if (
                    len(source_beads) != len(original_welds["beads"])
                    or set(source_beads) != expected_segments
                ):
                    raise MultiProbeVerificationError(
                        "original generated weld clusters do not match"
                    )
                report["timed_pose_welds"] = []
            for bead in batch["welds"]:
                labels = bead["stages"]
                durations = np.asarray(
                    [float(timing_by_label[label]["duration_s"]) for label in labels]
                )
                edges = np.r_[0.0, np.cumsum(durations)]
                spline = CubicSpline(
                    np.asarray(bead["times_s"]), np.asarray(bead["joint_states_7"]), axis=0
                )
                fixture = tuple(bead["fixture"])
                original_poses = [
                    pose(np.asarray(source_by_label[labels[0]]["states"][0], dtype=float))
                ]
                original_poses.extend(
                    pose(np.asarray(source_by_label[label]["states"][-1], dtype=float))
                    for label in labels
                )
                points = np.asarray([item[:3, 3] for item in original_poses])
                rotations = np.asarray([item[:3, :3] for item in original_poses])
                target_position, target_orientation, blend = build_local_weld_blend(
                    points, rotations, durations
                )
                actual_curves = [spline] * len(labels)
                if args.timed_pose_weld_batch:
                    source_bead = source_beads[bead["source_segment"]]
                    entry = {
                        "source_segment": bead["source_segment"],
                        "status": "in_progress",
                        "trials": [],
                    }
                    report["timed_pose_welds"].append(entry)
                    save()
                    try:
                        if (
                            source_bead["status"] != "sampled_exploratory_weld_only"
                            or [stage["stage"] for stage in source_bead["stages"]] != labels
                            or not np.array_equal(source_bead["fixture"], fixture)
                        ):
                            raise MultiProbeVerificationError(
                                "generated blend, fixture or stage order differs"
                            )
                        for mode in args.timed_pose_modes:
                            first, last = optimization_window(
                                [stage["kind"] for stage in source_bead["stages"]], mode
                            )
                            generated = {
                                **source_bead,
                                "times_s": bead["times_s"],
                                "joint_states_7": bead["joint_states_7"],
                                "fixture": bead["fixture"],
                                "stages": source_bead["stages"][first:last],
                            }
                            selected_blend = {
                                **blend,
                                "position_windows_s": blend["position_windows_s"][first:last],
                            }
                            for weight in args.timed_pose_weights:
                                trial = {"mode": mode, "weight": weight, "status": "in_progress"}
                                entry["trials"].append(trial)
                                save()

                                def save_weld(result):
                                    trial["result"] = result
                                    save()

                                try:
                                    with preserve_scene(env):
                                        result = correct_piecewise_weld(
                                            env,
                                            info,
                                            names,
                                            speeds,
                                            limits,
                                            checker,
                                            generated,
                                            [
                                                source_by_label[label]
                                                for label in labels[first:last]
                                            ],
                                            durations[first:last],
                                            points[first : last + 1],
                                            rotations[first : last + 1],
                                            lambda j, t, derivative=0: target_position(
                                                j + first, t, derivative
                                            ),
                                            lambda j, t: target_orientation(j + first, t),
                                            selected_blend,
                                            pose,
                                            save_weld,
                                            process_weight=PROCESS_WEIGHT,
                                            initial_trust_box=INITIAL_TRUST_BOX,
                                            constrain_process_knots=True,
                                            generated_lead_order=True,
                                            source_orientation_override=lambda j,
                                            t: target_orientation(j + first, t),
                                            coalesce_generated_samples=True,
                                            timed_pose_weight=weight,
                                            coalescing_fraction=args.coalescing_fraction,
                                            preview_only=args.coalescing_preview,
                                            clock_offset_s=float(edges[first]),
                                        )
                                    trial["result"] = result
                                    trial["status"] = result["status"]
                                    required_status = (
                                        "coalesced_preview_only"
                                        if args.coalescing_preview
                                        else "sampled_curve_process_acceptance_unresolved"
                                    )
                                    if result["status"] != required_status:
                                        raise MultiProbeVerificationError(
                                            "stage-local optimizer or actual checks failed"
                                        )
                                    selected_curves = stage_local_curves(
                                        result["stage_curves"],
                                        durations[first:last].tolist(),
                                        clock_start_s=float(edges[first]),
                                    )
                                    if not grid_mode:
                                        actual_curves[first:last] = selected_curves
                                        entry["result"] = result
                                except Exception as exc:
                                    trial.update(
                                        status="failed",
                                        error_type=type(exc).__name__,
                                        error=str(exc),
                                        traceback=traceback.format_exc(),
                                    )
                                save()
                        if grid_mode:
                            entry["status"] = "local_grid_complete"
                        elif entry["trials"][0]["status"] == (
                            "coalesced_preview_only"
                            if args.coalescing_preview
                            else "sampled_curve_process_acceptance_unresolved"
                        ):
                            entry["status"] = entry["trials"][0]["status"]
                        else:
                            entry.update(status="failed", error=entry["trials"][0].get("error"))
                    except Exception as exc:
                        entry.update(
                            status="failed",
                            error_type=type(exc).__name__,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                    save()
                    if entry["status"] == "failed":
                        continue
                for i, label in enumerate(labels):
                    if label in weld_labels:
                        raise MultiProbeVerificationError("duplicate exploratory weld stage")
                    replacements[label] = weld_stage(
                        label, float(durations[i]), actual_curves[i], float(edges[i]), fixture
                    )
                    curves[label] = actual_curves[i]
                    curve_offsets[label] = float(edges[i])
                    weld_targets[label] = (target_position, target_orientation, i)
                    weld_labels.add(label)
                    if 0 < i < len(labels) - 1:
                        source_bead_labels.add(label)
            if args.timed_pose_weld_batch:
                if grid_mode:
                    report["status"] = (
                        "local_grid_with_failures_not_a_whole_tour"
                        if any(
                            trial["status"] == "failed"
                            for entry in report["timed_pose_welds"]
                            for trial in entry["trials"]
                        )
                        else "local_grid_complete_not_a_whole_tour"
                    )
                    save()
                    if snapshot(scene) != original:
                        raise MultiProbeVerificationError("scene state changed")
                    return
                expected = (
                    "coalesced_preview_only"
                    if args.coalescing_preview
                    else "sampled_curve_process_acceptance_unresolved"
                )
                if any(entry["status"] != expected for entry in report["timed_pose_welds"]):
                    raise MultiProbeVerificationError("one or more timed-pose weld clusters failed")
                if args.coalescing_preview:
                    report["status"] = "coalesced_preview_only"
                    save()
                    if snapshot(scene) != original:
                        raise MultiProbeVerificationError("scene state changed")
                    return
        transit_rows = {row["stage"]: row for row in batch["transits"]}
        expected_transits = {row["stage"] for row in rows if row["stage"].startswith("transit:")}
        if len(transit_rows) != len(batch["transits"]) or set(transit_rows) != expected_transits:
            raise MultiProbeVerificationError("exactly one actual curve required per transit")
        for label, row in transit_rows.items():
            model = native.QuinticSplineModel.build(
                row["knots_s"],
                np.asarray(row["v0"]),
                np.asarray(row["a0"]),
                np.asarray(row["v1"]),
                np.asarray(row["a1"]),
            )
            replacements[label] = transit_stage(
                label, row["duration_s"], model, row["nodes"], tuple(row["fixture"])
            )
            curves[label] = build_jerk_model(
                row["knots_s"],
                np.asarray(row["v0"]),
                np.asarray(row["a0"]),
                np.asarray(row["v1"]),
                np.asarray(row["a1"]),
            ).curve(np.asarray(row["nodes"]))
        path, corners = assemble_actual_tour(rows, clock, replacements)
        report["unresolved_joint_linear_corners"] = corners
        report["ordered_stationary_events"] = [
            stage.label for stage in path if isinstance(stage, StationaryEvent)
        ]
        report["modeled_joins"] = [item._asdict() for item in diagnostic_joins(path)]
        report["outer_physical_derivatives"] = "unknown"
        report["modeled_duration_s"] = sum(
            float(stage.duration) for stage in path if isinstance(stage, ActualStage)
        )
        original_by_label = {row["stage"]: row for row in rows}
        for stage in path:
            if isinstance(stage, StationaryEvent):
                checked = inspect_sampled_curve(
                    checker, [stage.position], limits, SQP_FEASIBILITY_BAND
                )
                report["stages"].append(
                    {
                        "stage": stage.label,
                        "stationary_event": True,
                        "collision_and_position": checked,
                    }
                )
                save()
                continue
            ts = np.linspace(0, stage.duration, SAMPLES)
            states = [stage.evaluate(Seconds(float(t))) for t in ts]
            q = np.asarray([item.position for item in states])
            vel = np.asarray([item.velocity[:7] for item in states if item.velocity is not None])
            ratios = np.max(np.abs(vel), axis=0) / speeds
            checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            if stage.label in curves:
                extrema = (
                    cubic_position_extrema(curves[stage.label])
                    if stage.label in weld_labels
                    else position_extrema(curves[stage.label])
                )
                excess = [
                    max(limits[i, 0] - item["minimum"], item["maximum"] - limits[i, 1], 0.0)
                    for i, item in enumerate(extrema)
                ]
            else:
                # Source joint-linear and eased straight fixture stages are axis-wise bounded by their endpoints.
                source = np.asarray(original_by_label[stage.label]["states"], dtype=float)
                excess = np.max(
                    np.maximum(np.maximum(limits[:, 0] - source, source - limits[:, 1]), 0.0),
                    axis=0,
                ).tolist()
            if any(value > SQP_FEASIBILITY_BAND for value in excess):
                raise CurveLimitError(
                    f"{stage.label}: actual position extrema exceed configured band"
                )
            entry = {
                "stage": stage.label,
                "duration_s": float(stage.duration),
                "collision_and_position": checked,
                "raw_position_extrema_excess": excess,
                "sampled_speed_ratio": ratios.tolist(),
                "sampled_speeds_pass": bool(
                    np.all(ratios <= 1 + MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps)
                ),
            }
            if stage.label in curves:
                jerk = np.asarray(curves[stage.label](ts + curve_offsets.get(stage.label, 0.0), 3))
                entry["sampled_peak_jerk_by_axis"] = np.max(np.abs(jerk), axis=0).tolist()
                entry["sampled_integrated_squared_jerk_by_axis"] = np.trapezoid(
                    jerk**2, ts, axis=0
                ).tolist()
            if stage.label.startswith(("approach:", "search:", "retract:", "withdraw:", "weld:")):
                source = np.asarray(original_by_label[stage.label]["states"], dtype=float)
                first, last = pose(source[0]), pose(source[-1])
                poses = np.asarray([pose(point) for point in q])
                geometry = assess_process_geometry(first, last, poses)
                entry["geometry"] = geometry
                feed = np.linalg.norm(np.gradient(poses[:, :3, 3], ts, axis=0), axis=1)
                entry["modeled_feed_range_m_s"] = [float(feed.min()), float(feed.max())]
                if stage.label in phase_info:
                    phase = stage.label.split(":")[0]
                    source_duration = float(phase_info[stage.label]["source_duration_s"])
                    if phase in ("search", "retract"):
                        _, progress_clock = search_cruise_clock(source_duration)
                        u = progress_clock(ts)[0]
                        ramp = source_duration / 2
                        mask = (ts >= ramp) & (ts <= 2 * ramp)
                        entry["sampled_cruise_feed_range_m_s"] = [
                            float(feed[mask].min()),
                            float(feed[mask].max()),
                        ]
                    else:
                        u = one_sided_progress(
                            ts,
                            stage.duration,
                            1.0 if phase == "approach" else 0.0,
                            1.0 if phase == "withdraw" else 0.0,
                        )
                    expected_pos = first[:3, 3] + u[:, None] * (last[:3, 3] - first[:3, 3])
                    expected_rot = interpolate_rotations(first[:3, :3], last[:3, :3], u)
                    geometry["max_clock_position_error_m"] = float(
                        np.linalg.norm(poses[:, :3, 3] - expected_pos, axis=1).max()
                    )
                    geometry["max_clock_orientation_error_rad"] = float(
                        rotation_errors(expected_rot, poses[:, :3, :3]).max()
                    )
                    entry["source_nominal_feed_m_s"] = float(
                        np.linalg.norm(last[:3, 3] - first[:3, 3]) / source_duration
                    )
                if stage.label in weld_labels:
                    endpoint_pos = np.linalg.norm(
                        poses[[0, -1], :3, 3] - np.stack((first[:3, 3], last[:3, 3])), axis=1
                    )
                    endpoint_rot = rotation_errors(
                        np.stack((first[:3, :3], last[:3, :3])), poses[[0, -1], :3, :3]
                    )
                    entry["source_endpoint_position_error_m"] = endpoint_pos.tolist()
                    entry["source_endpoint_orientation_error_rad"] = endpoint_rot.tolist()
                    pos_band = (
                        CURVE_PARITY_ULPS
                        * np.finfo(float).eps
                        * max(
                            1.0,
                            float(np.linalg.norm(first[:3, 3])),
                            float(np.linalg.norm(last[:3, 3])),
                        )
                    )
                    rot_band = CURVE_PARITY_ULPS * np.finfo(float).eps
                    if np.any(endpoint_pos > pos_band) or np.any(endpoint_rot > rot_band):
                        raise MultiProbeVerificationError(
                            f"{stage.label}: source Cartesian anchor differs"
                        )
                    position_target, orientation_target, index = weld_targets[stage.label]
                    expected_pos = np.asarray([position_target(index, float(t)) for t in ts])
                    expected_rot = np.asarray([orientation_target(index, float(t)) for t in ts])
                    entry["blend_target_position_error_m"] = float(
                        np.linalg.norm(poses[:, :3, 3] - expected_pos, axis=1).max()
                    )
                    entry["blend_target_orientation_error_rad"] = float(
                        rotation_errors(expected_rot, poses[:, :3, :3]).max()
                    )
                else:
                    endpoint = np.max(np.abs(q[[0, -1]] - source[[0, -1]]), axis=0)
                    entry["original_endpoint_raw_residual_by_axis"] = endpoint.tolist()
                    if np.any(endpoint > SQP_FEASIBILITY_BAND):
                        raise MultiProbeVerificationError(
                            f"{stage.label}: generated source endpoint differs"
                        )
            report["stages"].append(entry)
            save()
        # Existing same-curve binary64 parity band; not a physical tolerance.
        band = CURVE_PARITY_ULPS * np.finfo(float).eps
        bad = []
        for join in report["modeled_joins"]:
            for field in ("position_delta", "velocity_delta", "acceleration_delta"):
                values = join[field]
                if values is None or max(map(abs, values), default=0) > band:
                    bad.append(
                        {
                            "left": join["left"],
                            "right": join["right"],
                            "field": field,
                            "raw_delta": values,
                        }
                    )
        report["unresolved_modeled_joins"] = bad
        bead_order_unresolved = [
            item["stage"]
            for item in report["stages"]
            if item["stage"] in source_bead_labels
            and item["geometry"]["max_backward_distance_m"]
            > CURVE_PARITY_ULPS * np.finfo(float).eps
        ]
        report["unresolved_source_bead_order"] = bead_order_unresolved
        if not all(item.get("sampled_speeds_pass", True) for item in report["stages"]):
            report["status"] = "failed_sampled_sourced_speed"
        elif bead_order_unresolved:
            report["status"] = "failed_source_bead_order"
        elif bad or corners:
            report["status"] = "failed_unresolved_modeled_joins_or_corners"
        else:
            report["status"] = "sampled_tour_physical_and_process_acceptance_unresolved"
        report["fixture_speed_and_acceleration_limits"] = "not supplied"
        report["search_hit_assumption"] = (
            "hypothetical hit at commanded limit; no observed controller event"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != original:
        raise MultiProbeVerificationError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "stages_checked": len(report["stages"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
