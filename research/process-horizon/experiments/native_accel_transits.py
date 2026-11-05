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
# Endpoint equality arithmetic only; not a physical tolerance.
SEED_ROUNDOFF_ULPS = 128


class TransitContractError(RuntimeError):
    """The generated stages cannot supply a modeled fixed-fixture spline."""


def fourth_jerk_weights(speeds, risk_weight, reference_time_s):
    """Return per-axis weights for a dimensionless soft fourth-power jerk risk.

    Reference time is in seconds; speeds use native axis units per second.
    A zero weight keeps the original squared-jerk-only problem.
    """
    speed = np.asarray(speeds, dtype=float)
    if speed.ndim != 1 or not np.all(np.isfinite(speed) & (speed > 0)):
        raise TransitContractError("finite positive sourced speeds required")
    if not np.isfinite(risk_weight) or risk_weight < 0:
        raise TransitContractError("jerk risk weight must be finite and nonnegative")
    if risk_weight == 0:
        return None
    if reference_time_s is None or not np.isfinite(reference_time_s) or reference_time_s <= 0:
        raise TransitContractError("positive finite risk reference time in seconds required")
    return JERK_MULTIPLIER * risk_weight * reference_time_s**4 / speed**4


def peak_jerk_configuration(speeds, risk_weight, reference_time_s):
    """Return dimensionless jerk normalization and soft peak-cost coefficients.

    Reference time is seconds and speeds use native axis units per second.
    A zero weight preserves the original problem and introduces no peak variable.
    """
    speed = np.asarray(speeds, dtype=float)
    if speed.ndim != 1 or not np.all(np.isfinite(speed) & (speed > 0)):
        raise TransitContractError("finite positive sourced speeds required")
    if not np.isfinite(risk_weight) or risk_weight < 0:
        raise TransitContractError("peak risk weight must be finite and nonnegative")
    if risk_weight == 0:
        return None
    if reference_time_s is None or not np.isfinite(reference_time_s) or reference_time_s <= 0:
        raise TransitContractError("positive finite peak reference time in seconds required")
    return reference_time_s**2 / speed, np.full(
        len(speed), JERK_MULTIPLIER * risk_weight / reference_time_s**3
    )


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
    acceleration_resolver=None,
    fourth_risk_weight=0.0,
    risk_reference_time_s=None,
    peak_risk_weight=0.0,
    peak_reference_time_s=None,
    peak_witness_times_s=None,
    seed_positions=None,
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
    fourth_weights = fourth_jerk_weights(speed, fourth_risk_weight, risk_reference_time_s)
    peak_config = peak_jerk_configuration(speed, peak_risk_weight, peak_reference_time_s)
    if seed_positions is not None and sum(s["stage"].startswith("transit:") for s in segments) != 1:
        raise TransitContractError("explicit seed requires exactly one transit")
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
            if seed_positions is not None:
                supplied = np.asarray(seed_positions, dtype=float)
                if supplied.shape != seed.shape or not np.isfinite(supplied).all():
                    raise TransitContractError(
                        "explicit transit seed must be finite seven-axis knots"
                    )
                if not np.allclose(
                    supplied[[0, -1]],
                    seed[[0, -1]],
                    rtol=0,
                    atol=SEED_ROUNDOFF_ULPS
                    * np.finfo(float).eps
                    * max(1.0, float(np.max(np.abs(seed)))),
                ):
                    raise TransitContractError("explicit transit seed changes endpoints")
                seed = supplied.copy()
                seed[[0, -1]] = current[[0, -1], :7]
            v0, v1 = left[:7], right[:7]
            if acceleration_resolver is None:
                raise TransitContractError("explicit nine-axis acceleration boundaries required")
            a0, a1 = acceleration_resolver(index)
            a0, a1 = np.asarray(a0, dtype=float), np.asarray(a1, dtype=float)
            if (
                a0.shape != (9,)
                or a1.shape != (9,)
                or not np.isfinite(a0).all()
                or not np.isfinite(a1).all()
                or np.any(a0[7:] != 0)
                or np.any(a1[7:] != 0)
            ):
                raise TransitContractError("transit fixture accelerations must be zero")
            duration = durations[index]
            knots_s = np.linspace(0, duration, KNOTS)
            model = native.QuinticSplineModel.build(knots_s.tolist(), v0, a0[:7], v1, a1[:7])
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
            if peak_config is not None:
                normalization, peak_cost = peak_config
                witnesses = (
                    knots_s
                    if peak_witness_times_s is None
                    else np.asarray(peak_witness_times_s, dtype=float)
                )
                if (
                    witnesses.ndim != 1
                    or len(witnesses) == 0
                    or not np.isfinite(witnesses).all()
                    or np.any(np.diff(witnesses) <= 0)
                    or witnesses[0] < 0
                    or witnesses[-1] > duration
                ):
                    raise TransitContractError("ordered in-domain peak witnesses required")
                initial_jerk = model.evaluate(np.asarray(witnesses), seed, 3)[3]
                initial_peak = np.max(np.abs(initial_jerk) * normalization, axis=0)
                peak_node = ti.Node("native-transit-peak")
                peak_node.addVar(
                    "peak", list(names[:7]), initial_peak, [ti.Bounds(0.0, float("inf"))] * 7
                )
                nodes.append(peak_node)
            variables = ti.NodesVariables("native-transit", nodes)
            knots = list(variables.getNodes())[:KNOTS]
            positions = [node.getVar("position") for node in knots]
            peak_var = variables.getNodes()[-1].getVar("peak") if peak_config is not None else None
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
            if fourth_weights is not None:
                problem.addCostSet(
                    native.QuinticSplineJerkFourthCost(model, positions, fourth_weights),
                    sqp.CostPenaltyType.SQUARED,
                )
            if peak_config is not None:
                problem.addConstraintSet(
                    native.QuinticSplineJerkPeakConstraint(
                        model, positions, peak_var, witnesses.tolist(), normalization
                    )
                )
                problem.addCostSet(
                    ti.JointPosConstraint(np.zeros(7), peak_var, peak_cost, "SoftJerkPeak"),
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
            q = (
                np.asarray(variables.getValues()).reshape(KNOTS, 7)
                if peak_var is None
                else np.vstack([position.value() for position in positions])
            )
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
                    "a0": a0[:7].tolist(),
                    "a1": a1[:7].tolist(),
                    "fixture": fixture.tolist(),
                    "input_jerk": jerk(seed),
                    "result_jerk": jerk(q),
                    **(
                        {
                            "peak_epigraph": np.asarray(peak_var.value()).tolist(),
                            "peak_witnesses_s": np.asarray(witnesses).tolist(),
                        }
                        if peak_var is not None
                        else {}
                    ),
                }
            )
    finally:
        _restore_scene(env, old, revision)
    return results
