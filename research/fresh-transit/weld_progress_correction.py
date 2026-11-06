"""Bounded native timed-progress correction of a generated weld cluster."""

from time import perf_counter

import _quintic_spline as native
import numpy as np
from _spline_dynamic import DynamicSplineCollisionConstraint
from actual_stage_adapters import weld_stage
from actual_stage_contract import diagnostic_joins
from continuous_spline_jerk import build_jerk_model
from free_space_transit import _ifopt_profile
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import rotation_errors
from scipy.interpolate import CubicSpline
from spline_velocity_extrema import velocity_extrema
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp
from tesseract_robotics.tesseract_common import Isometry3d
from weld_progress_grid import witness_grid
from weld_source_orientation import build_source_bead_orientation
from weld_target_progress import measure_target_progress

# One shared bounded experimental setting; no per-tour or per-stage tuning.
KNOTS_PER_STAGE = 5
WITNESSES_PER_STAGE = 9
SAMPLES_PER_STAGE = 1025
JERK_WEIGHT = 1.0
TIMED_PROGRESS_WEIGHT = 1.0
BLEND_LINE_WEIGHT = 1.0
SOLVE_TIME_S = 20.0
MAX_SQP_ITERATIONS = 200
MERIT_COEFFICIENT = 100.0


class WeldProgressCorrectionError(RuntimeError):
    """A generated weld cluster cannot form or validate the timed-progress solve."""


def correct_weld_progress(
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
    saved_iterate=None,
):
    """Refine actual shared q/qd/qdd against source-derived timed progress.

    Seven optimized axes are track in m and six arm axes in rad. Fixture axes
    are held at the generated positioner rad/tailstock m configuration. The
    clock is modeled seconds. Outer q/qd/qdd are from the generated curve, not
    measured controller boundaries. Every generated source waypoint remains an
    anchored endpoint. The local blend is exploratory and needs process approval.
    """
    fixture = np.asarray(saved["fixture"], dtype=float)
    if fixture.shape != (2,) or len(run) != len(durations) or len(points) != len(run) + 1:
        raise WeldProgressCorrectionError("generated weld cluster shape mismatch")
    edges, witness_groups = witness_grid(durations, WITNESSES_PER_STAGE)
    _, knot_groups = witness_grid(durations, KNOTS_PER_STAGE)
    knot_times = np.unique(np.concatenate(knot_groups))
    baseline = CubicSpline(
        np.asarray(saved["times_s"]), np.asarray(saved["joint_states_7"]), axis=0
    )
    generated_knots = baseline(knot_times)
    v0, a0 = baseline(0.0, 1), baseline(0.0, 2)
    v1, a1 = baseline(float(edges[-1]), 1), baseline(float(edges[-1]), 2)
    model = native.QuinticSplineModel.build(knot_times.tolist(), v0, a0, v1, a1)
    group = env.getJointGroup(info.manipulator)
    arm_limits = np.asarray(group.getLimits().joint_limits)
    env.setState(dict(zip(names[7:], map(float, fixture), strict=True)))
    nodes = []
    for index, q in enumerate(generated_knots):
        node = ti.Node(f"weld-timed-knot-{index}")
        bounds = (
            [ti.Bounds(float(value), float(value)) for value in q]
            if index in (0, len(generated_knots) - 1)
            else [ti.Bounds(float(low), float(high)) for low, high in arm_limits]
        )
        node.addVar("position", list(names[:7]), q, bounds)
        nodes.append(node)
    variables = ti.NodesVariables("generated-weld-progress", nodes)
    positions = [node.getVar("position") for node in variables.getNodes()]
    problem = sqp.TrajOptQPProblem(variables)
    problem.addCostSet(
        native.QuinticSplineJerkCost(model, positions, JERK_WEIGHT / np.asarray(speed) ** 2),
        sqp.CostPenaltyType.SQUARED,
    )
    evaluator = ti.SingleTimestepCollisionEvaluator(
        group, env, _ifopt_profile(True).collision_constraint_config, False
    )
    dynamic_constraints = []
    source_orientation = build_source_bead_orientation(rotations, durations, orientation)
    unique_witnesses = np.unique(np.concatenate(witness_groups))
    problem.addConstraintSet(
        native.QuinticSplinePositionLimits(
            model,
            positions,
            unique_witnesses.tolist(),
            [ti.Bounds(float(a), float(b)) for a, b in arm_limits],
        )
    )
    for j, times in enumerate(witness_groups):
        first = np.eye(4)
        first[:3, 3] = points[j]
        first[:3, :3] = rotations[j]
        last = np.eye(4)
        last[:3, 3] = points[j + 1]
        last[:3, :3] = rotations[j + 1]
        line = ti.CartLineInfo(
            group, "nozzle_link", "workpiece_link", Isometry3d(first), Isometry3d(last)
        )
        mapping = model.map(times.tolist(), 0)
        delta = points[j + 1] - points[j]
        length_squared = float(delta @ delta)
        if length_squared <= 0:
            raise WeldProgressCorrectionError("zero-length source process segment")
        local = times - edges[j]
        targets = np.asarray(
            [float((target(j, float(t)) - points[j]) @ delta / length_squared) for t in local]
        )
        problem.addConstraintSet(
            native.SplineCartesianProgressConstraint(
                line, positions, mapping.basis, mapping.offset, True, True, f"weld-order-{j}"
            )
        )
        problem.addCostSet(
            native.SplineTimedProgress(
                line,
                positions,
                mapping.basis,
                mapping.offset,
                targets,
                np.full(len(times), TIMED_PROGRESS_WEIGHT),
                f"weld-clock-{j}",
            ),
            sqp.CostPenaltyType.SQUARED,
        )
        for k, t in enumerate(local):
            term = native.SplineCartesianLineConstraint(
                line,
                positions,
                mapping.basis[k],
                mapping.offset[k],
                np.full(6, BLEND_LINE_WEIGHT),
                f"weld-line-{j}-{k}",
            )
            inside_blend = any(a < t < b for a, b in blend["position_windows_s"][j])
            if inside_blend:
                problem.addCostSet(term, sqp.CostPenaltyType.SQUARED)
            else:
                problem.addConstraintSet(term)
            collision_constraint = DynamicSplineCollisionConstraint(
                evaluator, positions, mapping.basis[k], mapping.offset[k], f"weld-collision-{j}-{k}"
            )
            dynamic_constraints.append(collision_constraint)
            problem.addConstraintSet(collision_constraint)
    problem.setup()

    def residual_snapshot():
        for constraint in dynamic_constraints:
            constraint.update()
        names = problem.getNLPConstraintNames()
        values = np.asarray(problem.getExactConstraintViolations().raw, dtype=float)
        if len(names) != len(values) or not np.isfinite(values).all():
            raise WeldProgressCorrectionError("constraint diagnostic names or residuals differ")
        return {
            "exact_cost": float(problem.getTotalExactCost()),
            "constraint_violations": dict(zip(names, map(float, values), strict=True)),
        }

    variables.setVariables(generated_knots.ravel())
    initial_residuals = residual_snapshot()
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    params = solver.params
    params.max_iter = MAX_SQP_ITERATIONS
    params.max_time = SOLVE_TIME_S
    params.cnt_tolerance = SQP_FEASIBILITY_BAND
    params.initial_merit_error_coeff = MERIT_COEFFICIENT
    solver.params = params
    started = perf_counter()
    if saved_iterate is None:
        solver.solve(problem)
        solver_status = str(solver.getStatus())
    else:
        stored_times = np.asarray(saved_iterate["knots_s"], dtype=float)
        stored_nodes = np.asarray(saved_iterate["nodes"], dtype=float)
        if (
            not np.array_equal(stored_times, knot_times)
            or stored_nodes.shape != generated_knots.shape
        ):
            raise WeldProgressCorrectionError("saved failed iterate uses different modeled knots")
        for name, actual in (("v0", v0), ("a0", a0), ("v1", v1), ("a1", a1), ("fixture", fixture)):
            if not np.array_equal(np.asarray(saved_iterate[name]), actual):
                raise WeldProgressCorrectionError(
                    "saved failed iterate uses different modeled boundaries"
                )
        variables.setVariables(stored_nodes.ravel())
        solver_status = saved_iterate["solver_status"]
    solver_s = perf_counter() - started
    nodes = np.vstack([p.value() for p in positions])
    result = {
        "status": "in_progress",
        "solver_status": solver_status,
        "solver_s_contended": solver_s,
        "knots_s": knot_times.tolist(),
        "nodes": nodes.tolist(),
        "v0": v0.tolist(),
        "a0": a0.tolist(),
        "v1": v1.tolist(),
        "a1": a1.tolist(),
        "fixture": fixture.tolist(),
        "stages": [],
        "initial_residuals": initial_residuals,
        "candidate_residuals": residual_snapshot(),
        "diagnostic_replay": saved_iterate is not None,
    }
    on_solved(result)
    curve = build_jerk_model(knot_times, v0, a0, v1, a1).curve(nodes)
    all_extrema = position_extrema(curve)
    position_excess = [
        max(arm_limits[k, 0] - x["minimum"], x["maximum"] - arm_limits[k, 1], 0.0)
        for k, x in enumerate(all_extrema)
    ]
    stages = [
        weld_stage(run[j]["stage"], float(durations[j]), curve, float(edges[j]), tuple(fixture))
        for j in range(len(run))
    ]
    result["joins"] = [item._asdict() for item in diagnostic_joins(stages)]
    failed = any(value > SQP_FEASIBILITY_BAND for value in position_excess)
    for j, stage in enumerate(stages):
        local = np.linspace(0.0, float(durations[j]), SAMPLES_PER_STAGE)
        t = edges[j] + local
        q7, jerk = curve(t), curve(t, 3)
        q = np.column_stack((q7, np.tile(fixture, (len(t), 1))))
        try:
            collision = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
        except (CurveCollisionError, CurveLimitError) as exc:
            collision = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            failed = True
        speed_extrema = velocity_extrema(curve, float(edges[j]), float(edges[j + 1]))
        speed_excess = [
            max(abs(x["minimum"]) - speed[k], abs(x["maximum"]) - speed[k], 0.0)
            for k, x in enumerate(speed_extrema)
        ]
        failed |= any(value > SQP_FEASIBILITY_BAND for value in speed_excess)
        actual = np.asarray([pose(value) for value in q])
        first = np.eye(4)
        first[:3, 3] = points[j]
        first[:3, :3] = rotations[j]
        last = np.eye(4)
        last[:3, 3] = points[j + 1]
        last[:3, :3] = rotations[j + 1]
        geometry = assess_process_geometry(first, last, actual)
        targets = np.asarray([target(j, float(t)) for t in local])
        target_rot = np.asarray([source_orientation(j, float(t)) for t in local])
        feed = np.linalg.norm(np.gradient(actual[:, :3, 3], local, axis=0), axis=1)
        result["stages"].append(
            {
                "stage": stage.label,
                "geometry": geometry,
                "target_progress_3d": measure_target_progress(actual[:, :3, 3], targets, local),
                "source_start_position_error_m": float(
                    np.linalg.norm(actual[0, :3, 3] - points[j])
                ),
                "source_end_position_error_m": float(
                    np.linalg.norm(actual[-1, :3, 3] - points[j + 1])
                ),
                "collision": collision,
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
                "speed_extrema_excess": speed_excess,
                "peak_sampled_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                "integrated_normalized_jerk": float(
                    np.trapezoid(np.sum((jerk / speed) ** 2, axis=1), local)
                ),
            }
        )
    result["position_extrema_excess"] = position_excess
    result["status"] = (
        "solver_failed_actual_curve_diagnostic"
        if solver_status != "SQPStatus.NLP_CONVERGED"
        else "failed_actual_extrema"
        if failed
        else "sampled_curve_process_acceptance_unresolved"
    )
    return result
