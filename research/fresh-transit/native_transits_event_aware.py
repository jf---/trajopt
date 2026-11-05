"""Opt-in native spline refinement of freshly generated transit stages.

The existing joint-linear clock provides modeled boundary slopes; this module
makes no claim about controller time or process-stage continuity. The caller
must independently check the assembled actual spline and process trajectories.
"""

from time import perf_counter

import _quintic_spline as native
import numpy as np
from _spline_dynamic import DynamicSplineCollisionConstraint
from free_space_transit import _ifopt_profile, _restore_scene
from fresh_boundaries import modeled_boundaries
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp

KNOTS = 7
COLLOCATION_POINTS = 33
LVS_METERS = 0.025
JERK_MULTIPLIER = 1e-6


class TransitContractError(RuntimeError):
    """The generated stages cannot supply a modeled fixed-fixture spline."""


def refine_transits(
    env,
    info,
    names,
    segments,
    clock,
    speeds,
    *,
    start_velocity,
    end_velocity,
    boundary_resolver=modeled_boundaries,
):
    """Return independent transit spline outcomes, without changing process stages.

    Axis order is track in metres, six robot axes in radians, positioner in
    radians and tailstock in metres. Speeds and clock are per second. The first
    and last boundary velocities are supplied explicitly by the caller. Actual
    physical endpoint velocities remain unspecified.
    """
    if len(names) != 9 or len(set(names)) != 9 or len(segments) != len(clock["stages"]):
        raise TransitContractError("nine axes and matching stage clock required")
    if any(
        stage["stage"] != timing.get("stage")
        for stage, timing in zip(segments, clock["stages"], strict=True)
    ):
        raise TransitContractError("stage and clock labels mismatch")
    states = [np.asarray(stage["states"], dtype=float) for stage in segments]
    durations = [float(stage["duration_s"]) for stage in clock["stages"]]
    speed = np.asarray(speeds, dtype=float)
    if speed.shape != (7,) or not np.all(np.isfinite(speed) & (speed > 0)):
        raise TransitContractError("seven finite positive sourced speeds required")
    if any(not np.isfinite(t) or t < 0 for t in durations):
        raise TransitContractError("nonnegative modeled durations required")
    if any(
        x.ndim != 2 or x.shape[1] != 9 or len(x) < 2 or not np.isfinite(x).all() for x in states
    ):
        raise TransitContractError("finite nine-axis stages required")
    if any(not np.array_equal(a[-1], b[0]) for a, b in zip(states[:-1], states[1:], strict=True)):
        raise TransitContractError("nine-axis stage join mismatch")
    old = {str(k): float(v) for k, v in env.getState().joints.items()}
    revision = int(env.getRevision())
    results = []
    try:
        for index, stage in enumerate(segments):
            if not stage["stage"].startswith("transit:"):
                continue
            current = states[index]
            fixture = current[0, 7:]
            if not np.all(current[:, 7:] == fixture):
                raise TransitContractError("transit changes fixture")
            left, right, events = boundary_resolver(
                segments,
                clock["stages"],
                index,
                start_velocity=start_velocity,
                end_velocity=end_velocity,
            )
            original = np.linspace(0.0, 1.0, len(current))
            target = np.linspace(0.0, 1.0, KNOTS)
            seed = np.column_stack([np.interp(target, original, current[:, i]) for i in range(7)])
            v0, v1 = left[:7], right[:7]
            duration = durations[index]
            knots_s = np.linspace(0, duration, KNOTS)
            model = native.QuinticSplineModel.build(
                knots_s.tolist(), v0, np.zeros(7), v1, np.zeros(7)
            )
            env.setState(dict(zip(names[-2:], map(float, fixture), strict=True)))
            limits = np.asarray(env.getJointGroup(info.manipulator).getLimits().joint_limits)
            nodes = []
            for j, q in enumerate(seed):
                node = ti.Node(f"native-transit-{j}")
                bounds = (
                    [ti.Bounds(float(x), float(x)) for x in q]
                    if j in (0, KNOTS - 1)
                    else [ti.Bounds(float(a), float(b)) for a, b in limits]
                )
                node.addVar("position", list(names[:7]), q, bounds)
                nodes.append(node)
            variables = ti.NodesVariables("native-transit", nodes)
            knots = list(variables.getNodes())
            positions = [node.getVar("position") for node in knots]
            profile = _ifopt_profile(True)
            cfg = profile.collision_cost_config
            check = cfg.collision_check_config
            check.longest_valid_segment_length = LVS_METERS
            cfg.collision_check_config = check
            profile.collision_cost_config = cfg
            profile.smooth_velocities = profile.smooth_accelerations = profile.smooth_jerks = False
            terms = profile.create(info, env, knots, [0, KNOTS - 1])
            problem = sqp.TrajOptQPProblem(variables)
            for c in terms.constraints:
                problem.addConstraintSet(c)
            for collection, penalty in (
                (terms.absolute_costs, sqp.CostPenaltyType.ABSOLUTE),
                (terms.squared_costs, sqp.CostPenaltyType.SQUARED),
                (terms.hinge_costs, sqp.CostPenaltyType.HINGE),
            ):
                for c in collection:
                    problem.addCostSet(c, penalty)
            problem.addCostSet(
                native.QuinticSplineJerkCost(model, positions, JERK_MULTIPLIER / speed**2),
                sqp.CostPenaltyType.SQUARED,
            )
            samples = np.linspace(0, duration, COLLOCATION_POINTS + 2)[1:-1].tolist()
            problem.addConstraintSet(
                native.QuinticSplinePositionLimits(
                    model, positions, samples, [ti.Bounds(float(a), float(b)) for a, b in limits]
                )
            )
            mapping = model.map(samples, 0)
            evaluator = ti.SingleTimestepCollisionEvaluator(
                env.getJointGroup(info.manipulator),
                env,
                _ifopt_profile(True).collision_constraint_config,
                False,
            )
            for j in range(len(samples)):
                problem.addConstraintSet(
                    DynamicSplineCollisionConstraint(
                        evaluator,
                        positions,
                        mapping.basis[j],
                        mapping.offset[j],
                        f"ActualSplineCollision_{j}",
                    )
                )
            problem.setup()
            qp = sqp.PIQPSolver()
            settings = qp.settings
            settings.eps_abs, settings.eps_rel, settings.check_duality_gap = 1e-7, 1e-9, True
            settings.verbose = True
            qp.settings = settings
            solver = sqp.TrustRegionSQPSolver(qp)
            params = solver.params
            (
                params.max_iter,
                params.max_time,
                params.max_merit_coeff_increases,
                params.cnt_tolerance,
            ) = 200, 10.0, 10, 1e-4
            solver.params = params
            started = perf_counter()
            solver.solve(problem)
            elapsed = perf_counter() - started
            status = str(solver.getStatus())
            if status != "SQPStatus.NLP_CONVERGED":
                raise TransitContractError(f"{stage['stage']}: {status}")
            q = np.asarray(variables.getValues()).reshape(KNOTS, 7)
            quad = model.jerk_quadrature()

            def jerk(x):
                y = quad.map.basis @ x + quad.map.offset
                return float(np.sum(np.asarray(quad.weights)[:, None] * y**2 / speed[None, :] ** 2))

            results.append(
                {
                    "stage": stage["stage"],
                    "preserved_event_indices": list(events),
                    "status": status,
                    "solver_s": elapsed,
                    "duration_s": duration,
                    "knots_s": knots_s.tolist(),
                    "nodes": q.tolist(),
                    "v0": v0.tolist(),
                    "v1": v1.tolist(),
                    "fixture": fixture.tolist(),
                    "input_jerk": jerk(seed),
                    "result_jerk": jerk(q),
                }
            )
    finally:
        _restore_scene(env, old, revision)
    return results
