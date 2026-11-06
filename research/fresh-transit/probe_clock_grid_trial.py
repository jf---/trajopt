"""Compare shared probe clocks and inspect a weld's actual 3D progress in one scene."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline  # noqa: F401  # Load the selected native runtime.
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401  # Check selected SQP runtime.
from actual_stage_contract import diagnostic_joins
from episode_refinement_samples import select_refinement_samples
from free_space_transit import _restore_scene
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from generated_knot_coalescing import SHORT_INTERVAL_FRACTION
from probe_clock_variants import search_cruise_clock_with_ramp
from probe_cruise_generate import generate_cruise_group
from probe_cruise_trial import one_sided_progress
from probe_process_inputs import extract_process_tour
from probe_source_join_trial import pose_matrix
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import interpolate_rotations, rotation_errors
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from spline_velocity_extrema import velocity_extrema
from weld_actual_witnesses import select_weld_witnesses
from weld_bead_block import bead_block_indices, representative_bead_block_indices
from weld_block_dynamics import block_dynamics
from weld_contact_audit import audit_bead_collision
from weld_local_blend import build_local_weld_blend
from weld_only_inputs import WeldBead, extract_tour
from weld_piecewise_correction import correct_piecewise_weld
from weld_progress_correction import correct_weld_progress
from weld_source_orientation import build_source_bead_orientation
from weld_tangent_leads import fit_lead
from weld_target_progress import measure_target_progress

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Independent finite actual-curve samples per process stage.
PARITY_ULPS = 32768  # Binary64 same-curve join check, not a physical tolerance.
WITNESS_CLOCK_ULPS = 256  # Binary64 modeled clock boundary arithmetic.
# A shared ablation, not a per-probe policy: baseline, shorter one-sided, shorter ramps.
SETTINGS = ((15.0 / 8.0, 0.5), (189.0 / 125.0, 0.5), (189.0 / 125.0, 0.25))
PROCESS_WEIGHT_GRID = (1.0, 10.0, 100.0)  # Shared exploratory cost coefficients.
TRUST_WEIGHT_GRID = (
    (0.01, 1.0),
    (0.1, 10.0),
    (0.01, 10.0),
)  # Mixed-unit SQP boxes, process weights.
REPRESENTATIVE_TRUST_BOX = 0.01  # Same mixed-unit SQP box as the prior contact retry.
REPRESENTATIVE_PROCESS_WEIGHT = 10.0  # Same process-cost coefficient as the prior contact retry.


class ProbeGridError(RuntimeError):
    """Fresh process, common policy or actual curve cannot be checked."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "weld-batch",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--weld-only", action="store_true", help="audit saved weld without repeating probe grid"
    )
    parser.add_argument(
        "--weld-audit-only",
        action="store_true",
        help="inspect every generated process stage without solving",
    )
    parser.add_argument(
        "--weld-iterate", type=Path, help="diagnose a saved failed weld iterate without solving"
    )
    parser.add_argument(
        "--piecewise-weld", action="store_true", help="use exact stage-local cubic variables"
    )
    parser.add_argument(
        "--piecewise-without-clock-cost", action="store_true", help="diagnostic objective ablation"
    )
    parser.add_argument(
        "--piecewise-compare-clock", action="store_true", help="run both objectives in one scene"
    )
    parser.add_argument(
        "--piecewise-process-knots",
        action="store_true",
        help="add process constraints at existing knot times",
    )
    parser.add_argument(
        "--piecewise-generated-lead-order",
        action="store_true",
        help="use sampled generated-path ordering for leads",
    )
    parser.add_argument(
        "--piecewise-trust-weight-grid",
        action="store_true",
        help="compare three shared solver settings in one scene",
    )
    parser.add_argument(
        "--piecewise-bead-block",
        action="store_true",
        help="solve an unresolved bead and its immediate neighbors",
    )
    parser.add_argument("--piecewise-contact-audit-from", type=Path)
    parser.add_argument("--piecewise-contact-audit-index", type=int)
    parser.add_argument("--piecewise-contact-refine-from", type=Path)
    parser.add_argument("--piecewise-refine-trust-box", type=float)
    parser.add_argument("--piecewise-refine-process-weight", type=float)
    parser.add_argument("--piecewise-representative-segments", nargs="+", type=int)
    parser.add_argument("--piecewise-representative-contact-audit-from", type=Path)
    parser.add_argument("--piecewise-group-audit-from-segment", type=int)
    parser.add_argument("--piecewise-representative-retry-from", type=Path)
    parser.add_argument("--piecewise-qp-verbose", action="store_true")
    parser.add_argument("--piecewise-coalesce-generated", action="store_true")
    parser.add_argument("--piecewise-coalesce-preview", action="store_true")
    parser.add_argument(
        "--piecewise-coalescing-fraction", type=float, default=SHORT_INTERVAL_FRACTION
    )
    parser.add_argument("--piecewise-timed-pose-weights", nargs="+", type=float, default=[0.0])
    parser.add_argument(
        "--piecewise-adaptive-from",
        type=Path,
        help="select witnesses from a saved actual-curve diagnostic",
    )
    args = parser.parse_args()
    if args.piecewise_timed_pose_weights != [0.0] and not args.piecewise_representative_segments:
        raise ProbeGridError("timed pose grid requires representative process blocks")
    if any(not np.isfinite(weight) or weight < 0 for weight in args.piecewise_timed_pose_weights):
        raise ProbeGridError("timed pose grid weights must be finite and nonnegative")
    if len(set(args.piecewise_timed_pose_weights)) != len(args.piecewise_timed_pose_weights):
        raise ProbeGridError("timed pose grid weights must be distinct")
    if args.piecewise_weld and args.weld_iterate:
        raise ProbeGridError(
            "global-quintic iterates cannot initialize the stage-local cubic solve"
        )
    if (
        args.piecewise_without_clock_cost
        or args.piecewise_compare_clock
        or args.piecewise_adaptive_from
        or args.piecewise_process_knots
        or args.piecewise_generated_lead_order
        or args.piecewise_trust_weight_grid
        or args.piecewise_bead_block
        or args.piecewise_contact_audit_from
        or args.piecewise_contact_refine_from
        or args.piecewise_representative_segments
    ) and not args.piecewise_weld:
        raise ProbeGridError("clock-cost ablation requires the piecewise weld model")
    if args.piecewise_without_clock_cost and args.piecewise_compare_clock:
        raise ProbeGridError("choose one clock-cost experiment")
    if args.piecewise_trust_weight_grid and not (
        args.piecewise_process_knots and args.piecewise_generated_lead_order
    ):
        raise ProbeGridError("trust grid requires generated-lead order and existing process knots")
    if args.piecewise_trust_weight_grid and args.piecewise_adaptive_from:
        raise ProbeGridError("trust grid must use the unchanged existing-knot witness set")
    if args.piecewise_bead_block and not (
        args.piecewise_process_knots and args.piecewise_generated_lead_order
    ):
        raise ProbeGridError("bead block requires generated-lead order and existing process knots")
    if (args.piecewise_contact_audit_from is None) != (args.piecewise_contact_audit_index is None):
        raise ProbeGridError("contact audit requires both result path and candidate index")
    if args.piecewise_contact_refine_from and (
        not args.piecewise_bead_block
        or args.piecewise_refine_trust_box is None
        or args.piecewise_refine_process_weight is None
    ):
        raise ProbeGridError(
            "collision refinement requires a bead block and its source solver settings"
        )
    if args.piecewise_representative_segments and not (
        args.piecewise_process_knots and args.piecewise_generated_lead_order
    ):
        raise ProbeGridError("representative blocks require generated-lead order and process knots")
    if (
        args.piecewise_representative_contact_audit_from
        and not args.piecewise_representative_segments
    ):
        raise ProbeGridError("representative contact audit requires explicit source segments")
    if (
        args.piecewise_group_audit_from_segment is not None
        and not args.piecewise_representative_segments
    ):
        raise ProbeGridError("group state audit requires an explicit target segment")
    if args.piecewise_representative_retry_from and not args.piecewise_representative_segments:
        raise ProbeGridError("representative retry requires explicit source segments")
    if args.piecewise_coalesce_preview and not args.piecewise_coalesce_generated:
        raise ProbeGridError("coalesced preview requires the explicit coalescing mode")
    if args.piecewise_coalesce_generated and not args.piecewise_representative_segments:
        raise ProbeGridError("coalescing requires explicit representative source segments")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise ProbeGridError("fresh independently checked recovery required")
    if args.output.exists():
        raise ProbeGridError("output already exists")
    args.output.mkdir(parents=True)
    start = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - start
    original = snapshot(scene)
    env, program, source, tours, names, opw, limits7, _, _, home, info, _ = scene["inputs"]
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    tour = int(recovery["verified"]["tour"])
    rows, clock, speeds = prepare_fresh(
        env,
        program,
        source,
        tours,
        home,
        names,
        raw,
        metadata,
        tour,
        recovery["record"],
        recovery["verified"],
    )
    speed = np.asarray(speeds, dtype=float)
    limits = np.asarray(env.getJointGroup("probe_grid_nine", list(names)).getLimits().joint_limits)
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    labels = [row["stage"] for row in rows]
    approach = [i for i, label in enumerate(labels) if label.startswith("approach:")]
    selected = sorted({0, len(approach) // 2, len(approach) - 1})
    report = {
        "tour": tour,
        "setup_s_contended": setup,
        "selection": selected,
        "settings": SETTINGS,
        "scope": "Generated-only exploratory curves; physical boundaries, hit behavior and manufacturing acceptance unknown.",
        "cases": [],
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

    for factor, ramp in () if args.weld_only or args.weld_audit_only else SETTINGS:
        for ordinal in selected:
            index = approach[ordinal]
            phases = rows[index : index + 4]
            timing = clock["stages"][index : index + 4]
            item = {
                "probe": ordinal,
                "duration_factor": factor,
                "ramp_fraction": ramp,
                "status": "in_progress",
                "stages": [],
            }
            try:
                stages, evidence = generate_cruise_group(
                    env,
                    opw,
                    info,
                    names,
                    limits7,
                    speed,
                    phases,
                    timing,
                    one_sided_duration_factor=factor,
                    search_ramp_fraction=ramp,
                )
                item["joins"] = [j._asdict() for j in diagnostic_joins(stages)]
                for j, stage in enumerate(stages):
                    phase = evidence["process"][j]
                    points = np.asarray(phase["ik_states"], dtype=float)
                    duration = float(stage.duration)
                    knots = np.linspace(0.0, duration, len(points))
                    va = [np.asarray(phase[k]) for k in ("v0", "a0", "v1", "a1")]
                    curve = fit_lead(knots, points, (va[0], va[1]), (va[2], va[3]))
                    times = np.linspace(0.0, duration, SAMPLES)
                    q7 = curve(times)
                    fixture = np.asarray(phases[j]["states"][0][7:], dtype=float)
                    q = np.column_stack((q7, np.tile(fixture, (len(times), 1))))
                    collision = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
                    position = position_extrema(curve)
                    velocity = velocity_extrema(curve, 0.0, duration)
                    p_excess = [
                        max(limits[k, 0] - p["minimum"], p["maximum"] - limits[k, 1], 0.0)
                        for k, p in enumerate(position)
                    ]
                    v_excess = [
                        max(abs(p["minimum"]) - speed[k], abs(p["maximum"]) - speed[k], 0.0)
                        for k, p in enumerate(velocity)
                    ]
                    first, last = (
                        pose(np.asarray(phases[j]["states"])[0]),
                        pose(np.asarray(phases[j]["states"])[-1]),
                    )
                    tool = np.asarray([pose(point) for point in q])
                    geom = assess_process_geometry(first, last, tool)
                    feed = np.linalg.norm(np.gradient(tool[:, :3, 3], times, axis=0), axis=1)
                    if j in (1, 2):
                        _, target_clock = search_cruise_clock_with_ramp(
                            float(phase["source_duration_s"]), ramp_fraction=ramp
                        )
                        progress = target_clock(times)[0]
                    else:
                        progress = one_sided_progress(
                            times, duration, 1.0 if j == 0 else 0.0, 1.0 if j == 3 else 0.0
                        )
                    target_pos = first[:3, 3] + progress[:, None] * (last[:3, 3] - first[:3, 3])
                    target_rot = interpolate_rotations(first[:3, :3], last[:3, :3], progress)
                    jerk = np.asarray(curve(times, 3), dtype=float)
                    item["stages"].append(
                        {
                            "stage": stage.label,
                            "duration_s": duration,
                            "source_duration_s": phase["source_duration_s"],
                            "collision": collision,
                            "position_extrema_excess": p_excess,
                            "speed_extrema_excess": v_excess,
                            "geometry": geom,
                            "feed_range_m_s": [float(feed.min()), float(feed.max())],
                            "source_nominal_feed_m_s": phase["source_nominal_feed_m_s"],
                            "max_clock_position_error_m": float(
                                np.linalg.norm(tool[:, :3, 3] - target_pos, axis=1).max()
                            ),
                            "max_clock_orientation_error_rad": float(
                                rotation_errors(target_rot, tool[:, :3, :3]).max()
                            ),
                            "sampled_peak_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                            "sampled_integrated_normalized_jerk": float(
                                np.trapezoid(np.sum((jerk / speed) ** 2, axis=1), times)
                            ),
                        }
                    )
                join_band = PARITY_ULPS * np.finfo(float).eps
                joins_checked = all(
                    join[key] is not None and max(map(abs, join[key])) <= join_band
                    for join in item["joins"]
                    for key in ("position_delta", "velocity_delta", "acceleration_delta")
                )
                item["status"] = (
                    "sampled_curves_checked"
                    if joins_checked
                    and all(
                        max(s["position_extrema_excess"] + s["speed_extrema_excess"])
                        <= SQP_FEASIBILITY_BAND
                        for s in item["stages"]
                    )
                    else "failed_actual_extrema"
                )
            except Exception as exc:
                item.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            report["cases"].append(item)
            save()
            print(
                json.dumps(
                    {"probe": ordinal, "factor": factor, "ramp": ramp, "status": item["status"]}
                ),
                flush=True,
            )

    # Inspect closest progress along the actual 3D blend for the saved unresolved group.
    batch = json.loads(args.weld_batch.read_text())
    if args.piecewise_representative_segments:
        requested = set(args.piecewise_representative_segments)
        if len(requested) != len(args.piecewise_representative_segments):
            raise ProbeGridError("representative source segment list contains duplicates")
        unresolved = [b for b in batch["beads"] if b["source_segment"] in requested]
        if len(unresolved) != len(requested):
            raise ProbeGridError("a requested generated source segment is missing or duplicated")
        report["representative_blocks"] = []
    else:
        unresolved = [b for b in batch["beads"] if b["status"] == "sampled_bead_order_unresolved"]
    report["weld_audit"] = []
    task = extract_tour(source, tours, tour, "workpiece_link", home=home)
    process = extract_process_tour(program, task, probe_metadata_frame="workpiece_link")
    beads = {item.segment: item for item in process.operations if isinstance(item, WeldBead)}
    for saved, timed_pose_weight in (
        (item, weight) for item in unresolved for weight in args.piecewise_timed_pose_weights
    ):
        bead = beads[saved["source_segment"]]
        source_poses = np.asarray([pose_matrix(item) for item in bead.poses])
        run = [rows[labels.index(s["stage"])] for s in saved["stages"]]
        durations = np.asarray(
            [float(clock["stages"][labels.index(r["stage"])]["duration_s"]) for r in run]
        )
        start_pose, end_pose = (
            pose(np.asarray(run[0]["states"])[0]),
            pose(np.asarray(run[-1]["states"])[-1]),
        )
        points = np.vstack((start_pose[:3, 3], source_poses[:, :3, 3], end_pose[:3, 3]))
        rotations = np.concatenate(
            (start_pose[None, :3, :3], source_poses[:, :3, :3], end_pose[None, :3, :3])
        )
        target, orientation, blend = build_local_weld_blend(points, rotations, durations)
        edges = np.r_[0.0, np.cumsum(durations)]
        curve = CubicSpline(
            np.asarray(saved["times_s"]), np.asarray(saved["joint_states_7"]), axis=0
        )
        fixture = np.asarray(saved["fixture"])
        for j, stage in enumerate(saved["stages"]):
            if (
                not args.weld_audit_only
                and stage["stage"] not in saved["unresolved_bead_order_stages"]
            ):
                continue
            t = np.linspace(0.0, durations[j], SAMPLES)
            actual = np.asarray([pose(np.r_[q, fixture])[:3, 3] for q in curve(edges[j] + t)])
            desired = np.asarray([target(j, float(v)) for v in t])
            chord = points[j + 1] - points[j]
            progress = (actual - points[j]) @ chord / np.linalg.norm(chord)
            report["weld_audit"].append(
                {
                    "stage": stage["stage"],
                    "kind": stage["kind"],
                    "source_segment": saved["source_segment"],
                    "chord_backward_total_m": float(np.maximum(-np.diff(progress), 0).sum()),
                    "chord_backward_max_step_m": float(max(0.0, -np.diff(progress).min())),
                    "timed_target_max_error_m": float(
                        np.linalg.norm(actual - desired, axis=1).max()
                    ),
                    **measure_target_progress(actual, desired, t),
                }
            )
        if args.weld_audit_only:
            continue
        report["weld_correction"] = {"status": "in_progress"}
        old = {str(k): float(v) for k, v in env.getState().joints.items()}
        revision = int(env.getRevision())
        active_result = "weld_correction"
        try:

            def save_solved(result):
                report["weld_correction"] = result
                save()

            if args.piecewise_representative_segments:
                block = {
                    "source_segment": saved["source_segment"],
                    "timed_pose_weight": timed_pose_weight,
                    "status": "in_progress",
                    "attempts": [],
                }
                report["representative_blocks"].append(block)
                save()
                try:
                    if args.piecewise_group_audit_from_segment is not None:
                        previous = [
                            entry
                            for entry in batch["beads"]
                            if entry["source_segment"] == args.piecewise_group_audit_from_segment
                        ]
                        if len(previous) != 1:
                            raise ProbeGridError("one previous-fixture source segment is required")
                        before_fixture = np.asarray(previous[0]["fixture"], dtype=float)
                        env.setState(dict(zip(names[7:], map(float, before_fixture), strict=True)))
                        before_group = env.getJointGroup(info.manipulator)
                        env.setState(dict(zip(names[7:], map(float, fixture), strict=True)))
                        after_group = env.getJointGroup(info.manipulator)
                        q7 = np.asarray(saved["joint_states_7"], dtype=float)[
                            len(saved["joint_states_7"]) // 2
                        ]
                        actual = pose(np.r_[q7, fixture])

                        def group_relative(group):
                            fk = group.calcFwdKin(q7)
                            return np.linalg.solve(
                                np.asarray(fk["workpiece_link"].matrix),
                                np.asarray(fk["nozzle_link"].matrix),
                            )

                        old_pose, new_pose = (
                            group_relative(before_group),
                            group_relative(after_group),
                        )
                        block["group_state_audit"] = {
                            "before_source_segment": args.piecewise_group_audit_from_segment,
                            "before_fixture": before_fixture.tolist(),
                            "target_fixture": fixture.tolist(),
                            "before_group_position_error_m": float(
                                np.linalg.norm(old_pose[:3, 3] - actual[:3, 3])
                            ),
                            "after_group_position_error_m": float(
                                np.linalg.norm(new_pose[:3, 3] - actual[:3, 3])
                            ),
                            "before_group_orientation_error_rad": float(
                                rotation_errors(actual[None, :3, :3], old_pose[None, :3, :3])[0]
                            ),
                            "after_group_orientation_error_rad": float(
                                rotation_errors(actual[None, :3, :3], new_pose[None, :3, :3])[0]
                            ),
                        }
                        block["status"] = "kinematic_group_state_audit_only"
                        save()
                        continue
                    if args.piecewise_representative_contact_audit_from:
                        prior = json.loads(
                            args.piecewise_representative_contact_audit_from.read_text()
                        )
                        matching = [
                            entry
                            for entry in prior["representative_blocks"]
                            if entry["source_segment"] == saved["source_segment"]
                        ]
                        if len(matching) != 1 or not matching[0]["attempts"]:
                            raise ProbeGridError("one prior representative result is required")
                        candidate = matching[0]["attempts"][-1]
                        block["contact_coverage"] = []
                        for item, record in zip(
                            candidate["stages"], candidate["stage_curves"], strict=True
                        ):
                            if item["collision"].get("error_type") != "CurveLimitError":
                                continue
                            knots = np.asarray(record["knots_s"])
                            current = CubicSpline(
                                knots,
                                np.asarray(record["nodes"]),
                                axis=0,
                                bc_type=((1, record["v0"]), (1, record["v1"])),
                            )
                            times = np.linspace(knots[0], knots[-1], SAMPLES)
                            contacts = []
                            for index, time in enumerate(times):
                                for hit in checker.contacts(np.r_[current(time), fixture]):
                                    if float(hit.distance_m) < 0:
                                        contacts.append(
                                            {
                                                "sample": index,
                                                "time_s": float(time),
                                                "distance_m": float(hit.distance_m),
                                                "links": list(hit.links),
                                            }
                                        )
                            block["contact_coverage"].append(
                                {
                                    "stage": item["stage"],
                                    "sample_count": SAMPLES,
                                    "penetrating_samples": len(
                                        {item["sample"] for item in contacts}
                                    ),
                                    "worst_contact": min(
                                        contacts, key=lambda item: item["distance_m"]
                                    )
                                    if contacts
                                    else None,
                                    "contacts": contacts,
                                }
                            )
                        block["status"] = "independent_contact_coverage_only"
                        save()
                        continue
                    left, right = representative_bead_block_indices(saved["stages"])
                    block_saved = {**saved, "stages": saved["stages"][left:right]}
                    block_blend = {
                        **blend,
                        "position_windows_s": blend["position_windows_s"][left:right],
                    }
                    full_orientation = build_source_bead_orientation(
                        rotations, durations, orientation
                    )
                    block["stages"] = [s["stage"] for s in saved["stages"][left:right]]
                    block["generated_process"] = saved["stages"][left:right]

                    def run_representative(witnesses):
                        def checkpoint(result):
                            block["active_attempt"] = result
                            save()

                        result = correct_piecewise_weld(
                            env,
                            info,
                            names,
                            speed,
                            limits,
                            checker,
                            block_saved,
                            run[left:right],
                            durations[left:right],
                            points[left : right + 1],
                            rotations[left : right + 1],
                            lambda j, t, derivative=0: target(j + left, t, derivative),
                            lambda j, t: orientation(j + left, t),
                            block_blend,
                            pose,
                            checkpoint,
                            process_weight=REPRESENTATIVE_PROCESS_WEIGHT,
                            initial_trust_box=REPRESENTATIVE_TRUST_BOX,
                            constrain_process_knots=True,
                            generated_lead_order=True,
                            clock_offset_s=float(edges[left]),
                            source_orientation_override=lambda j, t: full_orientation(j + left, t),
                            collision_only_witness_times=witnesses,
                            qp_verbose=args.piecewise_qp_verbose,
                            coalesce_generated_samples=args.piecewise_coalesce_generated,
                            preview_only=args.piecewise_coalesce_preview,
                            timed_pose_weight=timed_pose_weight,
                            coalescing_fraction=args.piecewise_coalescing_fraction,
                        )
                        block["attempts"].append(result)
                        block["active_attempt"] = None
                        save()
                        return result

                    if args.piecewise_representative_retry_from:
                        prior = json.loads(args.piecewise_representative_retry_from.read_text())
                        matches = [
                            item
                            for item in prior["representative_blocks"]
                            if item["source_segment"] == saved["source_segment"]
                        ]
                        if len(matches) != 1 or not matches[0]["collision_witnesses"]:
                            raise ProbeGridError(
                                "one previously sampled collision witness set required"
                            )
                        previous_witnesses = {
                            int(index): values
                            for index, values in matches[0]["collision_witnesses"].items()
                        }
                        block["replayed_collision_witnesses"] = previous_witnesses
                        first = run_representative(previous_witnesses)
                    else:
                        first = run_representative(None)
                    collision_witnesses = {}
                    block["contact_audits"] = []
                    for j, item in enumerate(first["stages"]):
                        if item["collision"].get("error_type") != "CurveCollisionError":
                            continue
                        audit = audit_bead_collision(
                            env, info, names, checker, saved, first, item["stage"], SAMPLES
                        )
                        block["contact_audits"].append(audit)
                        times = np.linspace(edges[left + j], edges[left + j + 1], SAMPLES)
                        clearance = np.zeros(SAMPLES)
                        for contact in audit["all_contacts"]:
                            sample = int(contact["sample"])
                            if float(contact["time_s"]) != times[sample]:
                                raise ProbeGridError(
                                    "contact clock differs from the generated block"
                                )
                            clearance[sample] = min(clearance[sample], float(contact["distance_m"]))
                        collision_witnesses[j] = list(
                            select_refinement_samples(
                                times,
                                clearance,
                                np.zeros((SAMPLES, 9)),
                                audit["optimizer_collision_times_s"],
                                np.zeros(9),
                            )
                        )
                    block["collision_witnesses"] = {
                        str(j): values for j, values in collision_witnesses.items()
                    }
                    final = (
                        run_representative(collision_witnesses)
                        if collision_witnesses
                        and not args.piecewise_representative_retry_from
                        and not args.piecewise_coalesce_preview
                        else first
                    )
                    baseline_curves, final_curves = [], []
                    for record in final["stage_curves"]:
                        knots = np.asarray(record["knots_s"])
                        baseline_curves.append(
                            CubicSpline(
                                knots,
                                curve(knots),
                                axis=0,
                                bc_type=((1, curve(knots[0], 1)), (1, curve(knots[-1], 1))),
                            )
                        )
                        final_curves.append(
                            CubicSpline(
                                knots,
                                np.asarray(record["nodes"]),
                                axis=0,
                                bc_type=((1, record["v0"]), (1, record["v1"])),
                            )
                        )
                    block["generated_dynamics"] = block_dynamics(baseline_curves, speed)
                    block["final_dynamics"] = block_dynamics(final_curves, speed)
                    block["status"] = "evaluated_process_acceptance_unresolved"
                except Exception as exc:
                    block.update(
                        status="failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                save()
                continue

            if args.piecewise_contact_audit_from:
                previous = json.loads(args.piecewise_contact_audit_from.read_text())
                candidates = previous["block_setting_results"]
                index = args.piecewise_contact_audit_index
                if (
                    index < 0
                    or index >= len(candidates)
                    or len(saved["unresolved_bead_order_stages"]) != 1
                ):
                    raise ProbeGridError("contact audit candidate or unresolved bead is ambiguous")
                report["contact_audit"] = audit_bead_collision(
                    env,
                    info,
                    names,
                    checker,
                    saved,
                    candidates[index],
                    saved["unresolved_bead_order_stages"][0],
                    SAMPLES,
                )
                report["weld_correction"] = {"status": "contact_audit_only"}
                save()
                continue

            if args.piecewise_bead_block:
                left, right = bead_block_indices(
                    saved["stages"], saved["unresolved_bead_order_stages"]
                )
                block_saved = {**saved, "stages": saved["stages"][left:right]}
                block_blend = {
                    **blend,
                    "position_windows_s": blend["position_windows_s"][left:right],
                }
                full_orientation = build_source_bead_orientation(rotations, durations, orientation)
                report["block_scope"] = {
                    "stages": [stage["stage"] for stage in run[left:right]],
                    "outer_clock_s": [float(edges[left]), float(edges[right])],
                    "outside_stages_unchanged": [
                        stage["stage"] for stage in run[:left] + run[right:]
                    ],
                    "baseline_stages": saved["stages"],
                }
                collision_only_witnesses = None
                if args.piecewise_contact_refine_from:
                    audit = json.loads(args.piecewise_contact_refine_from.read_text())[
                        "contact_audit"
                    ]
                    indices = [
                        i
                        for i, item in enumerate(run[left:right])
                        if item["stage"] == audit["stage"]
                    ]
                    if len(indices) != 1:
                        raise ProbeGridError(
                            "collision audit stage is not unique in the bead block"
                        )
                    index = indices[0]
                    times = np.linspace(
                        edges[left + index], edges[left + index + 1], audit["sample_count"]
                    )
                    clearance = np.zeros(len(times))
                    for contact in audit["all_contacts"]:
                        sample = int(contact["sample"])
                        if float(contact["time_s"]) != times[sample]:
                            raise ProbeGridError(
                                "contact audit clock differs from the generated block"
                            )
                        clearance[sample] = min(clearance[sample], float(contact["distance_m"]))
                    witnesses = select_refinement_samples(
                        times,
                        clearance,
                        np.zeros((len(times), 9)),
                        audit["optimizer_collision_times_s"],
                        np.zeros(9),
                    )
                    if not witnesses:
                        raise ProbeGridError("contact episode has no new witnesses")
                    collision_only_witnesses = {index: list(witnesses)}
                    report["collision_refinement"] = {
                        "stage": audit["stage"],
                        "witnesses_s": list(witnesses),
                    }
                    block_settings = (
                        (args.piecewise_refine_trust_box, args.piecewise_refine_process_weight),
                    )
                else:
                    block_settings = (
                        TRUST_WEIGHT_GRID if args.piecewise_trust_weight_grid else ((0.1, 1.0),)
                    )
                report["block_setting_results"] = []
                for trust_box, weight in block_settings:

                    def save_block_result(result):
                        report["active_grid_result"] = result
                        save()

                    candidate = correct_piecewise_weld(
                        env,
                        info,
                        names,
                        speed,
                        limits,
                        checker,
                        block_saved,
                        run[left:right],
                        durations[left:right],
                        points[left : right + 1],
                        rotations[left : right + 1],
                        lambda j, t, derivative=0: target(j + left, t, derivative),
                        lambda j, t: orientation(j + left, t),
                        block_blend,
                        pose,
                        save_block_result,
                        process_weight=weight,
                        initial_trust_box=trust_box,
                        constrain_process_knots=True,
                        generated_lead_order=True,
                        clock_offset_s=float(edges[left]),
                        source_orientation_override=lambda j, t: full_orientation(j + left, t),
                        collision_only_witness_times=collision_only_witnesses,
                    )
                    report["block_setting_results"].append(candidate)
                    report["active_grid_result"] = None
                    save()
                report["weld_correction"] = {"status": "see_block_setting_results"}
                save()
                continue

            if args.piecewise_trust_weight_grid:
                report["trust_weight_grid"] = []
                for trust_box, weight in TRUST_WEIGHT_GRID:

                    def save_trust_grid_result(result):
                        report["active_grid_result"] = result
                        save()

                    candidate = correct_piecewise_weld(
                        env,
                        info,
                        names,
                        speed,
                        limits,
                        checker,
                        saved,
                        run,
                        durations,
                        points,
                        rotations,
                        target,
                        orientation,
                        blend,
                        pose,
                        save_trust_grid_result,
                        process_weight=weight,
                        initial_trust_box=trust_box,
                        constrain_process_knots=True,
                        generated_lead_order=True,
                    )
                    report["trust_weight_grid"].append(candidate)
                    report["active_grid_result"] = None
                    save()
                continue

            if args.piecewise_adaptive_from:
                previous = json.loads(args.piecewise_adaptive_from.read_text())["weld_correction"]
                if [item["stage"] for item in previous["stages"]] != [
                    item["stage"] for item in run
                ]:
                    raise ProbeGridError("saved witness stages do not match the generated process")
                sample_times, errors, contacts, progress = [], [], [], []
                for j, item in enumerate(previous["stage_curves"]):
                    knots = np.asarray(item["knots_s"], dtype=float)
                    if not np.allclose(
                        [knots[0], knots[-1]],
                        edges[j : j + 2],
                        rtol=0,
                        atol=WITNESS_CLOCK_ULPS * np.finfo(float).eps * max(1.0, float(edges[-1])),
                    ):
                        raise ProbeGridError("saved witness clock does not match generated process")
                    candidate = CubicSpline(
                        knots,
                        np.asarray(item["nodes"]),
                        axis=0,
                        bc_type=((1, item["v0"]), (1, item["v1"])),
                    )
                    times = np.linspace(knots[0], knots[-1], SAMPLES)
                    states = np.column_stack((candidate(times), np.tile(fixture, (len(times), 1))))
                    actual = np.asarray([pose(q)[:3, 3] for q in states])
                    desired = np.asarray([target(j, float(t - edges[j])) for t in times])
                    distances = np.asarray(
                        [
                            min([float(hit.distance_m) for hit in checker.contacts(q)], default=0.0)
                            for q in states
                        ]
                    )
                    sample_times.append(times)
                    errors.append(np.linalg.norm(actual - desired, axis=1))
                    contacts.append(distances)
                    delta = points[j + 1] - points[j]
                    progress.append((actual - points[j]) @ delta / np.linalg.norm(delta))
                witnesses = select_weld_witnesses(sample_times, errors, contacts, progress)
                report["adaptive_witnesses"] = {str(k): v for k, v in witnesses.items()}
                report["adaptive_source_max_target_error_m"] = max(float(x.max()) for x in errors)
                report["adaptive_source_min_contact_m"] = min(float(x.min()) for x in contacts)
                report["adaptive_source_max_reverse_step_m"] = min(
                    float(np.diff(x).min()) for x in progress
                )
                report["adaptive_selected_samples"] = [
                    {
                        "stage": run[j]["stage"],
                        "modeled_time_s": time,
                        "target_error_m": float(
                            errors[j][int(np.argmin(np.abs(sample_times[j] - time)))]
                        ),
                        "contact_distance_m": float(
                            contacts[j][int(np.argmin(np.abs(sample_times[j] - time)))]
                        ),
                    }
                    for j, times in witnesses.items()
                    for time in times
                ]
                report["weld_weight_grid"] = []
                for weight in (1.0,) if args.piecewise_process_knots else PROCESS_WEIGHT_GRID:

                    def save_grid_result(result):
                        report["active_grid_result"] = result
                        save()

                    candidate = correct_piecewise_weld(
                        env,
                        info,
                        names,
                        speed,
                        limits,
                        checker,
                        saved,
                        run,
                        durations,
                        points,
                        rotations,
                        target,
                        orientation,
                        blend,
                        pose,
                        save_grid_result,
                        extra_witness_times=witnesses,
                        process_weight=weight,
                        constrain_process_knots=args.piecewise_process_knots,
                        generated_lead_order=args.piecewise_generated_lead_order,
                    )
                    report["weld_weight_grid"].append(candidate)
                    report["active_grid_result"] = None
                    save()
                continue
            solver = correct_piecewise_weld if args.piecewise_weld else correct_weld_progress
            replay = (
                {
                    "clock_cost": not args.piecewise_without_clock_cost,
                    "constrain_process_knots": args.piecewise_process_knots,
                    "generated_lead_order": args.piecewise_generated_lead_order,
                }
                if args.piecewise_weld
                else {
                    "saved_iterate": None
                    if args.weld_iterate is None
                    else json.loads(args.weld_iterate.read_text())["weld_correction"]
                }
            )
            report["weld_correction"] = solver(
                env,
                info,
                names,
                speed,
                limits,
                checker,
                saved,
                run,
                durations,
                points,
                rotations,
                target,
                orientation,
                blend,
                pose,
                save_solved,
                **replay,
            )
            save()
            if args.piecewise_compare_clock:
                active_result = "weld_clock_ablation"
                report[active_result] = {"status": "in_progress"}

                def save_ablation(result):
                    report["weld_clock_ablation"] = result
                    save()

                report["weld_clock_ablation"] = correct_piecewise_weld(
                    env,
                    info,
                    names,
                    speed,
                    limits,
                    checker,
                    saved,
                    run,
                    durations,
                    points,
                    rotations,
                    target,
                    orientation,
                    blend,
                    pose,
                    save_ablation,
                    clock_cost=False,
                )
        except Exception as exc:
            report[active_result].update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            _restore_scene(env, old, revision)
        save()
    if snapshot(scene) != original:
        raise ProbeGridError("scene state changed")
    report["status"] = (
        "weld_diagnostic_only"
        if args.weld_only or args.weld_audit_only
        else (
            "completed_with_failures"
            if any(c["status"] != "sampled_curves_checked" for c in report["cases"])
            else "sampled_grid_checked"
        )
    )
    save()
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["cases"]),
                "weld_audits": len(report["weld_audit"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
