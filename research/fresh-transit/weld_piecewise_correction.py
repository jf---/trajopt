"""Stage-local cubic process refinement with native shared-derivative constraints."""

from time import perf_counter

import _quintic_spline as native
import numpy as np
from _spline_dynamic import DynamicSplineCollisionConstraint
from actual_stage_adapters import weld_stage
from actual_stage_contract import diagnostic_joins
from cubic_position_extrema import cubic_position_extrema
from cubic_velocity_extrema import cubic_velocity_extrema
from fixture_group_order import fixture_joint_group
from free_space_transit import _ifopt_profile
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve
from generated_knot_coalescing import SHORT_INTERVAL_FRACTION, coalesce_generated_knots
from lead_path_progress import sampled_path_progress
from piecewise_join_map import combine_join_maps
from piecewise_sqp_diagnostic import PiecewiseSqpTrace
from process_geometry import assess_process_geometry
from rotation_targets import rotation_errors
from scipy.interpolate import CubicSpline
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp
from tesseract_robotics.tesseract_common import Isometry3d
from weld_progress_grid import local_process_times, process_knot_witnesses, witness_grid
from weld_source_orientation import build_source_bead_orientation
from weld_target_progress import measure_target_progress

SAMPLES_PER_STAGE = 1025  # Same independent finite curve-check policy.
WITNESSES_PER_STAGE = 9  # Bounded common process/collision constraint grid.
CLOCK_ULPS = 256  # Binary64 modeled clock boundary arithmetic.
JERK_WEIGHT = 1.0
TIMED_PROGRESS_WEIGHT = 1.0
BLEND_LINE_WEIGHT = 1.0
ORIENTATION_NORMALIZATION_RAD = 1.0  # Objective scale in radians, not a process tolerance.
SOLVE_TIME_S = 20.0
MAX_SQP_ITERATIONS = 200
MERIT_COEFFICIENT = 100.0
DEFAULT_TRUST_BOX = 0.1  # Native mixed-axis SQP coordinates, not a physical distance.


class PiecewiseWeldError(RuntimeError):
    """A generated stage or its native actual-curve optimization is invalid."""


def correct_piecewise_weld(
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
    on_solved,
    clock_cost=True,
    extra_witness_times=None,
    process_weight=1.0,
    constrain_process_knots=False,
    initial_trust_box=DEFAULT_TRUST_BOX,
    generated_lead_order=False,
    clock_offset_s=0.0,
    source_orientation_override=None,
    collision_only_witness_times=None,
    qp_verbose=False,
    coalesce_generated_samples=False,
    preview_only=False,
    timed_pose_weight=0.0,
    timed_velocity_weight=0.0,
    coalescing_fraction=SHORT_INTERVAL_FRACTION,
    extra_callback=None,
):
    """Refine exact generated cubic stages with shared q/qd/qdd constraints.

    Axis order is track metres followed by six arm radians
    fixture remains at
    generated positioner radians/tailstock metres. Clock and outer derivatives
    are modeled, not physical controller observations. Collision is finitely
    sampled
    source-blend and feed acceptance are unresolved.
    """
    fixture = np.asarray(saved["fixture"], dtype=float)
    if fixture.shape != (2,) or len(run) != len(durations) or len(points) != len(run) + 1:
        raise PiecewiseWeldError("generated weld cluster shape mismatch")
    edges, witness_groups = witness_grid(durations, WITNESSES_PER_STAGE)
    if not np.isfinite(clock_offset_s):
        raise PiecewiseWeldError("modeled clock offset must be finite")
    edges += clock_offset_s
    witness_groups = [group + clock_offset_s for group in witness_groups]
    if not np.isfinite(process_weight) or process_weight <= 0:
        raise PiecewiseWeldError("process objective weight must be positive and finite")
    if not np.isfinite(timed_pose_weight) or timed_pose_weight < 0:
        raise PiecewiseWeldError("timed pose weight must be nonnegative and finite")
    if not np.isfinite(timed_velocity_weight) or timed_velocity_weight < 0:
        raise PiecewiseWeldError("timed velocity weight must be nonnegative and finite")
    if not np.isfinite(initial_trust_box) or initial_trust_box <= 0:
        raise PiecewiseWeldError("initial trust box must be positive and finite")
    if extra_witness_times is not None:
        for index, additions in extra_witness_times.items():
            values = np.asarray(additions, dtype=float)
            if (
                index < 0
                or index >= len(run)
                or values.ndim != 1
                or not np.isfinite(values).all()
                or np.any(values < edges[index])
                or np.any(values > edges[index + 1])
            ):
                raise PiecewiseWeldError("witness time outside its process stage")
            witness_groups[index] = np.unique(np.r_[witness_groups[index], values])
    original_times = np.asarray(saved["times_s"], dtype=float)
    baseline = CubicSpline(original_times, np.asarray(saved["joint_states_7"], dtype=float), axis=0)
    group = fixture_joint_group(env, info.manipulator, names[7:], fixture)
    arm_limits = np.asarray(group.getLimits().joint_limits)
    nodes, layouts = [], []
    for j in range(len(run)):
        left, right = float(edges[j]), float(edges[j + 1])
        band = CLOCK_ULPS * np.finfo(float).eps * max(1.0, abs(left), abs(right))
        knots = np.r_[
            left,
            original_times[(original_times > left + band) & (original_times < right - band)],
            right,
        ]
        removed = []
        if coalesce_generated_samples:
            protected = [left, right]
            protected.extend(
                left + float(t) for window in blend["position_windows_s"][j] for t in window
            )
            protected.extend(
                float(item["modeled_time_s"])
                for item in saved["witnesses"]
                if left <= float(item["modeled_time_s"]) <= right
            )
            knots, removed = coalesce_generated_knots(
                knots,
                baseline(knots),
                np.asarray(speed),
                protected,
                short_interval_fraction=coalescing_fraction,
            )
        model = native.CubicSplineStage.build(knots.tolist(), 7)
        initial = np.vstack((baseline(knots), baseline(left, 1), baseline(right, 1)))
        first_node = len(nodes)
        for k, value in enumerate(initial):
            is_velocity = k >= len(knots)
            fixed = (j == 0 and k in (0, len(knots))) or (
                j == len(run) - 1 and k in (len(knots) - 1, len(knots) + 1)
            )
            if fixed:
                bounds = [ti.Bounds(float(v), float(v)) for v in value]
            elif is_velocity:
                bounds = [ti.Bounds(-float(v), float(v)) for v in speed]
            else:
                bounds = [ti.Bounds(float(lo), float(hi)) for lo, hi in arm_limits]
            node = ti.Node(f"piecewise-{j}-{k}")
            node.addVar("value", list(names[:7]), value, bounds)
            nodes.append(node)
        layouts.append(
            {
                "model": model,
                "knots": knots,
                "initial": initial,
                "first_node": first_node,
                "count": len(initial),
                "removed_generated_samples_s": removed,
            }
        )
    variables = ti.NodesVariables("piecewise-weld", nodes)
    all_nodes = list(variables.getNodes())
    for layout in layouts:
        layout["variables"] = [
            n.getVar("value")
            for n in all_nodes[layout["first_node"] : layout["first_node"] + layout["count"]]
        ]
    problem = sqp.TrajOptQPProblem(variables)
    source_orientation = (
        build_source_bead_orientation(rotations, durations, orientation)
        if source_orientation_override is None
        else source_orientation_override
    )
    evaluator = ti.SingleTimestepCollisionEvaluator(
        group, env, _ifopt_profile(True).collision_constraint_config, False
    )
    dynamics = []
    process_groups = []
    collision_groups = []
    for j, layout in enumerate(layouts):
        model, stage_vars, knots = layout["model"], layout["variables"], layout["knots"]
        mids, dt = (knots[:-1] + knots[1:]) / 2, np.diff(knots)
        jerk = model.map(mids.tolist(), 3)
        coefficients = np.concatenate(
            [float(span) * JERK_WEIGHT / np.asarray(speed) ** 2 for span in dt]
        )
        problem.addCostSet(
            native.AffineSplineConstraint(
                stage_vars,
                jerk.basis,
                jerk.offset,
                [ti.Bounds(0.0, 0.0)] * len(coefficients),
                coefficients,
                f"piecewise-jerk-{j}",
            ),
            sqp.CostPenaltyType.SQUARED,
        )
        times = witness_groups[j]
        mapping = model.map(times.tolist(), 0)
        collision_times = times
        if collision_only_witness_times is not None and j in collision_only_witness_times:
            additions = np.asarray(collision_only_witness_times[j], dtype=float)
            if (
                additions.ndim != 1
                or not np.isfinite(additions).all()
                or np.any(additions < edges[j])
                or np.any(additions > edges[j + 1])
            ):
                raise PiecewiseWeldError("collision witness time outside its process stage")
            collision_times = np.unique(np.r_[times, additions])
        collision_groups.append(collision_times)
        collision_mapping = model.map(collision_times.tolist(), 0)
        process_times = process_knot_witnesses(times, knots) if constrain_process_knots else times
        process_groups.append(process_times)
        process_mapping = model.map(process_times.tolist(), 0)
        lower = [ti.Bounds(float(lo), float("inf")) for _ in times for lo, _ in arm_limits]
        upper = [ti.Bounds(-float("inf"), float(hi)) for _ in times for _, hi in arm_limits]
        for name, bounds in (("lower", lower), ("upper", upper)):
            problem.addConstraintSet(
                native.AffineSplineConstraint(
                    stage_vars,
                    mapping.basis,
                    mapping.offset,
                    bounds,
                    np.ones(len(bounds)),
                    f"piecewise-limits-{name}-{j}",
                )
            )
        first = np.eye(4)
        first[:3, 3] = points[j]
        first[:3, :3] = rotations[j]
        last = np.eye(4)
        last[:3, 3] = points[j + 1]
        last[:3, :3] = rotations[j + 1]
        line = ti.CartLineInfo(
            group, "nozzle_link", "workpiece_link", Isometry3d(first), Isometry3d(last)
        )
        delta = points[j + 1] - points[j]
        length_squared = float(delta @ delta)
        if length_squared <= 0:
            raise PiecewiseWeldError("zero-length process segment")
        local = local_process_times(process_times, float(edges[j]), float(durations[j]))
        targets = np.asarray(
            [float((target(j, float(t)) - points[j]) @ delta / length_squared) for t in local]
        )
        if generated_lead_order and saved["stages"][j]["kind"] == "generated_lead":
            path_points = np.asarray([target(j, float(t)) for t in local])
            path_velocity = np.asarray([target(j, float(t), 1) for t in local])
            path_tangents, path_arc = sampled_path_progress(path_points, path_velocity)
            problem.addConstraintSet(
                native.SplineCartesianPathProgressConstraint(
                    line,
                    stage_vars,
                    process_mapping.basis,
                    process_mapping.offset,
                    path_points,
                    path_tangents,
                    path_arc,
                    True,
                    True,
                    f"piecewise-order-{j}",
                )
            )
        else:
            problem.addConstraintSet(
                native.SplineCartesianProgressConstraint(
                    line,
                    stage_vars,
                    process_mapping.basis,
                    process_mapping.offset,
                    True,
                    True,
                    f"piecewise-order-{j}",
                )
            )
        if clock_cost:
            problem.addCostSet(
                native.SplineTimedProgress(
                    line,
                    stage_vars,
                    process_mapping.basis,
                    process_mapping.offset,
                    targets,
                    np.full(len(process_times), TIMED_PROGRESS_WEIGHT * process_weight),
                    f"piecewise-clock-{j}",
                ),
                sqp.CostPenaltyType.SQUARED,
            )
        target_poses = []
        if timed_pose_weight > 0 or timed_velocity_weight > 0:
            for local_time in local:
                target_pose = np.eye(4)
                target_pose[:3, 3] = target(j, float(local_time))
                target_pose[:3, :3] = source_orientation(j, float(local_time))
                target_poses.append(target_pose)
        if timed_velocity_weight > 0:
            reference_speed = float(np.sqrt(length_squared) / durations[j])
            for k in range(len(local) - 1):
                problem.addCostSet(
                    native.SplineTimedCartesianVelocity(
                        line,
                        target_poses[k],
                        target_poses[k + 1],
                        stage_vars,
                        process_mapping.basis[k],
                        process_mapping.offset[k],
                        process_mapping.basis[k + 1],
                        process_mapping.offset[k + 1],
                        float(local[k + 1] - local[k]),
                        reference_speed,
                        float(durations[j]),
                        timed_velocity_weight,
                        f"piecewise-timed-velocity-{j}-{k}",
                    ),
                    sqp.CostPenaltyType.SQUARED,
                )
        for k, local_time in enumerate(local):
            if timed_pose_weight > 0:
                problem.addCostSet(
                    native.SplineTimedCartesianPose(
                        line,
                        target_poses[k],
                        stage_vars,
                        process_mapping.basis[k],
                        process_mapping.offset[k],
                        float(np.sqrt(length_squared)),
                        ORIENTATION_NORMALIZATION_RAD,
                        timed_pose_weight,
                        f"piecewise-timed-pose-{j}-{k}",
                    ),
                    sqp.CostPenaltyType.SQUARED,
                )
            term = native.SplineCartesianLineConstraint(
                line,
                stage_vars,
                process_mapping.basis[k],
                process_mapping.offset[k],
                np.full(
                    6,
                    BLEND_LINE_WEIGHT * process_weight
                    if any(a < local_time < b for a, b in blend["position_windows_s"][j])
                    else BLEND_LINE_WEIGHT,
                ),
                f"piecewise-line-{j}-{k}",
            )
            if any(a < local_time < b for a, b in blend["position_windows_s"][j]):
                problem.addCostSet(term, sqp.CostPenaltyType.SQUARED)
            else:
                problem.addConstraintSet(term)
        for k in range(len(collision_times)):
            collision = DynamicSplineCollisionConstraint(
                evaluator,
                stage_vars,
                collision_mapping.basis[k],
                collision_mapping.offset[k],
                f"piecewise-collision-{j}-{k}",
            )
            dynamics.append(collision)
            problem.addConstraintSet(collision)
    for j in range(len(layouts) - 1):
        left, right = layouts[j], layouts[j + 1]
        for order in range(3):
            left_map = left["model"].map([float(edges[j + 1])], order)
            right_map = right["model"].map([float(edges[j + 1])], order)
            join_basis = combine_join_maps(left_map.basis, right_map.basis)
            join_offset = left_map.offset - right_map.offset
            problem.addConstraintSet(
                native.AffineSplineConstraint(
                    left["variables"] + right["variables"],
                    join_basis,
                    join_offset,
                    [ti.Bounds(0.0, 0.0)] * 7,
                    np.ones(7),
                    f"piecewise-join-{j}-{order}",
                )
            )
    for side, layout, time in (
        (0, layouts[0], float(edges[0])),
        (1, layouts[-1], float(edges[-1])),
    ):
        mapping = layout["model"].map([time], 2)
        target_acceleration = baseline(time, 2)
        problem.addConstraintSet(
            native.AffineSplineConstraint(
                layout["variables"],
                mapping.basis,
                mapping.offset,
                [ti.Bounds(float(v), float(v)) for v in target_acceleration],
                np.ones(7),
                f"piecewise-outer-acceleration-{side}",
            )
        )
    problem.setup()
    initial_nodes = np.concatenate([layout["initial"].ravel() for layout in layouts])
    variables.setVariables(initial_nodes)
    for dynamic in dynamics:
        dynamic.update()
    initial_values = np.asarray(problem.getExactConstraintViolations().raw, dtype=float)
    initial_names = problem.getNLPConstraintNames()
    if len(initial_values) != len(initial_names):
        raise PiecewiseWeldError("initial constraint diagnostic mismatch")
    initial = {
        "exact_cost": float(problem.getTotalExactCost()),
        "costs": dict(
            zip(problem.getNLPCostNames(), map(float, problem.getExactCosts()), strict=True)
        ),
        "constraint_violations": dict(zip(initial_names, map(float, initial_values), strict=True)),
        "stage_knot_counts": [len(layout["knots"]) for layout in layouts],
    }
    qp_solver = sqp.PIQPSolver()
    qp_solver.verbosity = int(qp_verbose)
    solver = sqp.TrustRegionSQPSolver(qp_solver)
    params = solver.params
    params.max_iter = MAX_SQP_ITERATIONS
    params.max_time = SOLVE_TIME_S
    params.cnt_tolerance = SQP_FEASIBILITY_BAND
    params.initial_merit_error_coeff = MERIT_COEFFICIENT
    params.initial_trust_box_size = initial_trust_box
    solver.params = params
    observer = PiecewiseSqpTrace(dynamics)
    solver.registerCallback(observer)
    if extra_callback is not None:
        solver.registerCallback(extra_callback)
    started = perf_counter()
    if not preview_only:
        solver.solve(problem)
    elapsed = perf_counter() - started
    solver_status = "INITIAL_RECONSTRUCTION_ONLY" if preview_only else str(solver.getStatus())
    curves, stage_records = [], []
    for layout in layouts:
        values = np.vstack([value.value() for value in layout["variables"]])
        n = len(layout["knots"])
        curves.append(
            CubicSpline(
                layout["knots"], values[:n], axis=0, bc_type=((1, values[n]), (1, values[n + 1]))
            )
        )
        stage_records.append(
            {
                "knots_s": layout["knots"].tolist(),
                "nodes": values[:n].tolist(),
                "v0": values[n].tolist(),
                "v1": values[n + 1].tolist(),
            }
        )
    result = {
        "status": "in_progress",
        "solver_status": solver_status,
        "qp_status": str(qp_solver.getSolverStatus()),
        "qp_settings": {
            "eps_abs": float(qp_solver.settings.eps_abs),
            "eps_rel": float(qp_solver.settings.eps_rel),
            "max_iter": int(qp_solver.settings.max_iter),
            "verbose": qp_verbose,
        },
        "solver_s_contended": elapsed,
        "initial": initial,
        "clock_cost_enabled": clock_cost,
        "timed_pose_weight": timed_pose_weight,
        "timed_velocity_weight": timed_velocity_weight,
        "timed_velocity_scales": "stage chord divided by modeled stage duration in m/s; time normalized by stage duration",
        "coalescing_fraction": coalescing_fraction,
        "timed_pose_position_scale": "stage chord length in metres; objective normalization only",
        "timed_pose_orientation_scale_rad": ORIENTATION_NORMALIZATION_RAD,
        "process_weight": process_weight,
        "generated_lead_order": generated_lead_order,
        "modeled_clock_offset_s": clock_offset_s,
        "initial_trust_box_native_units": initial_trust_box,
        "witness_times_s": [times.tolist() for times in witness_groups],
        "process_witness_times_s": [times.tolist() for times in process_groups],
        "collision_witness_times_s": [times.tolist() for times in collision_groups],
        "trace": observer.records,
        "trace_scope": "Successful QP trials only; failed QPs do not invoke the native callback. Dynamic row counts reflect the last constraint evaluation.",
        "stage_curves": stage_records,
        "removed_generated_samples_s": [
            layout["removed_generated_samples_s"] for layout in layouts
        ],
        "stages": [],
    }
    if qp_verbose:
        qp_point = np.zeros(problem.getNumQPVars())
        best_point = np.asarray(variables.getValues(), dtype=float)
        qp_point[: len(best_point)] = best_point
        result["last_qp_at_best_with_zero_slacks"] = native.qp_diagnostic(problem, qp_point)
    on_solved(result)
    stages = [
        weld_stage(run[j]["stage"], float(durations[j]), curves[j], float(edges[j]), tuple(fixture))
        for j in range(len(run))
    ]
    result["joins"] = [item._asdict() for item in diagnostic_joins(stages)]
    result["initial_reconstruction_max_delta_by_order"] = []
    for layout in layouts:
        knots = layout["knots"]
        initial = layout["initial"]
        reconstruction = CubicSpline(
            knots,
            initial[: len(knots)],
            axis=0,
            bc_type=((1, initial[len(knots)]), (1, initial[len(knots) + 1])),
        )
        times = np.linspace(knots[0], knots[-1], SAMPLES_PER_STAGE)
        result["initial_reconstruction_max_delta_by_order"].append(
            {
                f"derivative_{order}": np.max(
                    np.abs(reconstruction(times, order) - baseline(times, order)), axis=0
                ).tolist()
                for order in range(3)
            }
        )
    result["outer_boundary_delta_from_generated"] = {
        side: {
            f"derivative_{order}": (curve(time, order) - baseline(time, order)).tolist()
            for order in range(3)
        }
        for side, curve, time in (
            ("start", curves[0], float(edges[0])),
            ("end", curves[-1], float(edges[-1])),
        )
    }
    failed = False
    for j, stage in enumerate(stages):
        local = np.linspace(0.0, float(durations[j]), SAMPLES_PER_STAGE)
        times = edges[j] + local
        q7, jerk = curves[j](times), curves[j](times, 3)
        q = np.column_stack((q7, np.tile(fixture, (len(times), 1))))
        try:
            collision = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
        except (CurveCollisionError, CurveLimitError) as exc:
            collision = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            failed = True
        pos = cubic_position_extrema(curves[j])
        vel = cubic_velocity_extrema(curves[j])
        p_excess = [
            max(arm_limits[k, 0] - p["minimum"], p["maximum"] - arm_limits[k, 1], 0.0)
            for k, p in enumerate(pos)
        ]
        v_excess = [
            max(abs(v["minimum"]) - speed[k], abs(v["maximum"]) - speed[k], 0.0)
            for k, v in enumerate(vel)
        ]
        failed |= any(value > SQP_FEASIBILITY_BAND for value in p_excess + v_excess)
        actual = np.asarray([pose(value) for value in q])
        targets = np.asarray([target(j, float(t)) for t in local])
        target_rot = np.asarray([source_orientation(j, float(t)) for t in local])
        first = np.eye(4)
        first[:3, 3] = points[j]
        first[:3, :3] = rotations[j]
        last = np.eye(4)
        last[:3, 3] = points[j + 1]
        last[:3, :3] = rotations[j + 1]
        feed = np.linalg.norm(np.gradient(actual[:, :3, 3], local, axis=0), axis=1)
        result["stages"].append(
            {
                "stage": stage.label,
                "collision": collision,
                "position_extrema_excess": p_excess,
                "speed_extrema_excess": v_excess,
                "geometry": assess_process_geometry(first, last, actual),
                "target_progress_3d": measure_target_progress(actual[:, :3, 3], targets, local),
                "source_start_position_error_m": float(
                    np.linalg.norm(actual[0, :3, 3] - points[j])
                ),
                "source_end_position_error_m": float(
                    np.linalg.norm(actual[-1, :3, 3] - points[j + 1])
                ),
                "max_target_position_error_m": float(
                    np.linalg.norm(actual[:, :3, 3] - targets, axis=1).max()
                ),
                "max_source_orientation_error_rad": float(
                    rotation_errors(target_rot, actual[:, :3, :3]).max()
                ),
                "feed_range_m_s": [float(feed.min()), float(feed.max())],
                "source_nominal_feed_m_s": float(
                    np.linalg.norm(points[j + 1] - points[j]) / durations[j]
                ),
                "peak_sampled_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                "integrated_normalized_jerk": float(
                    np.trapezoid(np.sum((jerk / speed) ** 2, axis=1), local)
                ),
            }
        )
        if coalesce_generated_samples:
            original_states = np.column_stack((baseline(times), np.tile(fixture, (len(times), 1))))
            original_pose = np.asarray([pose(value) for value in original_states])
            result["stages"][-1]["max_generated_fk_position_delta_m"] = float(
                np.linalg.norm(actual[:, :3, 3] - original_pose[:, :3, 3], axis=1).max()
            )
            result["stages"][-1]["max_generated_fk_orientation_delta_rad"] = float(
                rotation_errors(original_pose[:, :3, :3], actual[:, :3, :3]).max()
            )
    result["status"] = (
        "coalesced_preview_only"
        if preview_only
        else "solver_failed_actual_curve_diagnostic"
        if solver_status != "SQPStatus.NLP_CONVERGED"
        else "failed_actual_extrema"
        if failed
        else "sampled_curve_process_acceptance_unresolved"
    )
    return result
