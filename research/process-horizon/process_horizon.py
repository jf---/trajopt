"""Opt-in native coordinated process spline with Cartesian constraints."""

import _quintic_spline as native
import numpy as np
from progress_samples import progress_sample_times
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp

# Binary64 allowance when a previous solver returns a fixed equality value.
FIXED_AXIS_ROUNDOFF_ULPS = 32


class ProcessHorizonError(RuntimeError):
    """Spline input or ordered process geometry is inconsistent."""


def build_process_horizon(
    names,
    times,
    seed,
    limits,
    v0,
    a0,
    v1,
    a1,
    jerk_weights,
    legs,
    samples_per_leg,
    collision_factory,
    *,
    progress_witnesses=(),
    line_witnesses=(),
    line_mode="hard",
    fixed_axes=None,
    fixed_reversals=(),
):
    """Build one fixed-clock native Ifopt spline across ordered process legs.

    Each leg defines a native workpiece-relative Cartesian line and a closed
    modeled time interval. Line, endpoint and ordering constraints apply at
    actual spline samples. The caller supplies explicit collision constraints
    and must independently check the continuous model at additional times.
    """
    clock = np.asarray(times, dtype=float)
    states = np.array(seed, dtype=float, copy=True)
    bounds = np.asarray(limits, dtype=float)
    weights = np.asarray(jerk_weights, dtype=float)
    if (
        clock.ndim != 1
        or len(clock) < 2
        or not np.isfinite(clock).all()
        or np.any(np.diff(clock) <= 0)
        or states.shape != (len(clock), len(names))
        or not np.isfinite(states).all()
        or bounds.shape != (len(names), 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
        or np.any(states < bounds[:, 0])
        or np.any(states > bounds[:, 1])
        or weights.shape != (len(names),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or samples_per_leg < 2
        or not legs
    ):
        raise ProcessHorizonError(
            "finite clock, seed, joint bounds, weights and process legs required"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(legs)
        for index, _ in (*progress_witnesses, *line_witnesses)
    ):
        raise ProcessHorizonError("process witness refers to an unknown leg")
    if line_mode not in ("hard", "soft"):
        raise ProcessHorizonError("line mode must be explicitly hard or soft")
    fixed = {} if fixed_axes is None else dict(fixed_axes)
    for axis, value in fixed.items():
        if (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not 0 <= axis < len(names)
            or not np.isfinite(value)
            or not bounds[axis, 0] <= value <= bounds[axis, 1]
            or np.any(
                np.abs(states[:, axis] - value)
                > FIXED_AXIS_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, abs(value))
            )
            or any(np.asarray(x, dtype=float)[axis] != 0 for x in (v0, a0, v1, a1))
        ):
            raise ProcessHorizonError(
                "fixed axis requires matching seed and zero endpoint derivatives"
            )
        states[:, axis] = value
    previous = float(clock[0])
    for leg in legs:
        begin, end = float(leg["start_s"]), float(leg["end_s"])
        if begin != previous or not begin < end or end > clock[-1]:
            raise ProcessHorizonError("process legs must cover clock in order")
        previous = end
    if previous != clock[-1]:
        raise ProcessHorizonError("process legs must cover the entire clock")
    reversals = {}
    for index, position, gradient in fixed_reversals:
        position = np.asarray(position, dtype=float)
        gradient = np.asarray(gradient, dtype=float)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index <= 0
            or index >= len(clock) - 1
            or index in reversals
            or not any(float(leg["end_s"]) == clock[index] for leg in legs[:-1])
            or position.shape != (len(names),)
            or gradient.shape != (len(names),)
            or not np.isfinite(position).all()
            or not np.isfinite(gradient).all()
            or np.linalg.norm(gradient) == 0
            or np.any(position < bounds[:, 0])
            or np.any(position > bounds[:, 1])
            or any(position[axis] != value for axis, value in fixed.items())
        ):
            raise ProcessHorizonError(
                "fixed reversal requires a valid shared knot, generated position and gradient"
            )
        states[index] = position
        reversals[index] = gradient
    model = native.QuinticSplineModel.build(
        clock.tolist(),
        np.asarray(v0, dtype=float),
        np.asarray(a0, dtype=float),
        np.asarray(v1, dtype=float),
        np.asarray(a1, dtype=float),
    )
    nodes = []
    for i, state in enumerate(states):
        node = ti.Node(f"process-knot-{i}")
        node_bounds = (
            [ti.Bounds(float(x), float(x)) for x in state]
            if i in (0, len(states) - 1) or i in reversals
            else [ti.Bounds(float(a), float(b)) for a, b in bounds]
        )
        for axis, value in fixed.items():
            node_bounds[axis] = ti.Bounds(float(value), float(value))
        node.addVar("position", list(names), state, node_bounds)
        nodes.append(node)
    variables = ti.NodesVariables("process-horizon", nodes)
    knots = [node.getVar("position") for node in variables.getNodes()]
    problem = sqp.TrajOptQPProblem(variables)
    problem.addCostSet(
        native.QuinticSplineJerkCost(model, knots, weights), sqp.CostPenaltyType.SQUARED
    )
    for index, gradient in reversals.items():
        velocity_map = model.map([float(clock[index])], 1)
        problem.addConstraintSet(
            native.SplineFixedJoinProgressVelocity(
                knots,
                velocity_map.basis[0],
                velocity_map.offset[0],
                gradient,
                f"fixed-reversal-{index}",
            )
        )
    all_times = []
    for leg_index, leg in enumerate(legs):
        leg_times = np.linspace(float(leg["start_s"]), float(leg["end_s"]), samples_per_leg)
        line_times = progress_sample_times(
            leg_times, [time for index, time in line_witnesses if index == leg_index]
        )
        mapping = model.map(line_times.tolist(), 0)
        info = leg["line"]
        coefficients = np.ones(len(info.indices))
        for index in range(len(line_times)):
            line = native.SplineCartesianLineConstraint(
                info,
                knots,
                mapping.basis[index],
                mapping.offset[index],
                coefficients,
                f"line-{len(all_times)}-{index}",
            )
            if line_mode == "hard":
                problem.addConstraintSet(line)
            else:
                problem.addCostSet(line, sqp.CostPenaltyType.SQUARED)
        witnesses = [time for index, time in progress_witnesses if index == leg_index]
        progress_times = progress_sample_times(leg_times, witnesses)
        progress_map = model.map(progress_times.tolist(), 0)
        problem.addConstraintSet(
            native.SplineCartesianProgressConstraint(
                info,
                knots,
                progress_map.basis,
                progress_map.offset,
                True,
                True,
                f"progress-{len(all_times)}",
            )
        )
        all_times.extend(line_times.tolist())
    all_times = np.unique(np.asarray(all_times, dtype=float))
    for constraint in collision_factory(knots, model, all_times):
        problem.addConstraintSet(constraint)
    problem.setup()
    return variables, problem, model
