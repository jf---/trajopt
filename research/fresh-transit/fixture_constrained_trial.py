"""Bounded nine-axis fixture trajectory experiment on a generated recovery."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from _spline_dynamic import DynamicSplineCollisionConstraint
from continuous_spline_jerk import build_jerk_model
from fixture_multichord import build_fixture_multichord, fixture_knot_times
from fixture_velocity_bounds import fixture_velocity_bounds
from free_space_transit import _ifopt_profile
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries
from spline_velocity_extrema import velocity_extrema
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

COLLOCATION = 33  # Existing native actual-curve collision/spline baseline.
SAMPLES = 1025  # Existing independent actual-curve collision check per stage.
MAX_ROUNDS = 6  # Shared bounded derivative-extremum witness policy.
MAX_ITER = 200  # Existing native SQP profile.
MAX_TIME_S = 10.0  # Existing per-attempt SQP time budget, seconds.
JERK_MULTIPLIER = 1e-6  # Existing time-aware native objective scale.


class FixtureConstrainedError(RuntimeError):
    """A generated fixture path cannot be optimized and checked."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "recovery", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--chain", action="append", required=True, help="first fixture stage; selection only"
    )
    parser.add_argument(
        "--duration-factor",
        action="append",
        type=float,
        required=True,
        help="shared modeled duration factor",
    )
    parser.add_argument(
        "--jerk-scale",
        action="append",
        type=float,
        required=True,
        help="shared soft jerk objective scale",
    )
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in [*args.duration_factor, *args.jerk_scale]):
        raise FixtureConstrainedError("positive finite duration factors and jerk scales required")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise FixtureConstrainedError("independently verified generated recovery required")
    if args.output.exists():
        raise FixtureConstrainedError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    original = snapshot(scene)
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
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
    group = env.getJointGroup("fixture_constrained_nine", list(names))
    active_links = list(group.getActiveLinkNames())
    if "workpiece_link" not in active_links:
        raise FixtureConstrainedError(
            "moving workpiece must be in the nine-axis collision active links"
        )
    limits = np.asarray(group.getLimits().joint_limits)
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    report = {
        "tour": tour,
        "scene_setup_s_contended": setup,
        "active_links": active_links,
        "scope": "Nine-axis modeled fixture interpolation; sampled collision and bounded actual-velocity witnesses; no physical boundary or fixture dynamics claim.",
        "chains": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    labels = [r["stage"] for r in rows]
    for factor, jerk_scale, label in [
        (factor, jerk_scale, label)
        for factor in args.duration_factor
        for jerk_scale in args.jerk_scale
        for label in args.chain
    ]:
        result = {
            "first_stage": label,
            "duration_factor": factor,
            "jerk_scale": jerk_scale,
            "attempts": [],
            "status": "in_progress",
        }
        try:
            i = labels.index(label)
            if not label.startswith("fixture:") or (i and labels[i - 1].startswith("fixture:")):
                raise FixtureConstrainedError("selection must name the first fixture stage")
            end = i + 1
            while end < len(rows) and labels[end].startswith("fixture:"):
                end += 1
            chain = rows[i:end]
            durations = [factor * float(t["duration_s"]) for t in clock["stages"][i:end]]
            first, last = np.asarray(chain[0]["states"]), np.asarray(chain[-1]["states"])
            v0 = (first[-1] - first[0]) / durations[0]
            v1 = (last[-1] - last[0]) / durations[-1]
            v0[7:] = 0.0
            v1[7:] = 0.0
            a0 = np.zeros(9)
            a1 = np.zeros(9)
            model, seed, stage_times = build_fixture_multichord(chain, durations, v0, a0, v1, a1)
            _, knots_s = fixture_knot_times(durations)
            stage_states = np.asarray([chain[0]["states"][0], *[r["states"][-1] for r in chain]])
            # Fixture objective scaling is derived solely from modeled motion, not a physical speed limit.
            fixture_scale = (
                np.sum(np.abs(np.diff(stage_states[:, 7:], axis=0)), axis=0) / stage_times[-1]
            )
            weights = np.zeros(9)
            weights[:7] = JERK_MULTIPLIER / np.asarray(speeds) ** 2
            for axis in (7, 8):
                if fixture_scale[axis - 7] > 0:
                    weights[axis] = JERK_MULTIPLIER / fixture_scale[axis - 7] ** 2
                else:
                    # An exactly held axis is fixed at every knot; its cost is identically zero.
                    weights[axis] = JERK_MULTIPLIER
            weights *= jerk_scale
            samples = np.unique(
                np.r_[np.linspace(0.0, stage_times[-1], COLLOCATION + 2)[1:-1], stage_times]
            )
            result.update(
                {
                    "stage_labels": [r["stage"] for r in chain],
                    "modeled_boundary_v0": v0.tolist(),
                    "modeled_boundary_v1": v1.tolist(),
                    "modeled_duration_s": sum(durations),
                    "boundary_acceleration": "zero modeled hypothesis; physical derivative unknown",
                    "fixture_speed_limits": "not supplied",
                    "knots_s": knots_s.tolist(),
                }
            )
            for attempt in range(MAX_ROUNDS):
                nodes = []
                for j, q in enumerate(seed):
                    node = ti.Node(f"fixture-{j}")
                    fixed = any(float(knots_s[j]) == float(t) for t in stage_times)
                    node_bounds = (
                        [ti.Bounds(float(x), float(x)) for x in q]
                        if fixed
                        else [ti.Bounds(float(a), float(b)) for a, b in limits]
                    )
                    for axis in (7, 8):
                        if fixture_scale[axis - 7] == 0:
                            node_bounds[axis] = ti.Bounds(
                                float(stage_states[0, axis]), float(stage_states[0, axis])
                            )
                    node.addVar("position", list(names), q, node_bounds)
                    nodes.append(node)
                variables = ti.NodesVariables("fixture-knots", nodes)
                positions = [n.getVar("position") for n in variables.getNodes()]
                problem = sqp.TrajOptQPProblem(variables)
                problem.addCostSet(
                    native.QuinticSplineJerkCost(model, positions, weights),
                    sqp.CostPenaltyType.SQUARED,
                )
                bounds = fixture_velocity_bounds(stage_states, stage_times, samples, speeds)
                velocity_bounds = [ti.Bounds(float(a), float(b)) for row in bounds for a, b in row]
                problem.addConstraintSet(
                    native.QuinticSplineDerivativeLimits(
                        model, positions, samples.tolist(), velocity_bounds
                    )
                )
                problem.addConstraintSet(
                    native.QuinticSplinePositionLimits(
                        model,
                        positions,
                        samples.tolist(),
                        [ti.Bounds(float(a), float(b)) for a, b in limits],
                    )
                )
                mapping = model.map(samples.tolist(), 0)
                evaluator = ti.SingleTimestepCollisionEvaluator(
                    group, env, _ifopt_profile(True).collision_constraint_config, False
                )
                for j in range(len(samples)):
                    problem.addConstraintSet(
                        DynamicSplineCollisionConstraint(
                            evaluator,
                            positions,
                            mapping.basis[j],
                            mapping.offset[j],
                            f"fixture-collision-{j}",
                        )
                    )
                problem.setup()
                qp = sqp.PIQPSolver()
                settings = qp.settings
                settings.eps_abs, settings.eps_rel, settings.check_duality_gap, settings.verbose = (
                    1e-7,
                    1e-9,
                    True,
                    False,
                )
                qp.settings = settings
                solver = sqp.TrustRegionSQPSolver(qp)
                params = solver.params
                (
                    params.max_iter,
                    params.max_time,
                    params.max_merit_coeff_increases,
                    params.cnt_tolerance,
                ) = MAX_ITER, MAX_TIME_S, 10, SQP_FEASIBILITY_BAND
                solver.params = params
                began = perf_counter()
                solver.solve(problem)
                elapsed = perf_counter() - began
                status = str(solver.getStatus())
                solved = np.vstack([p.value() for p in positions])
                attempt_result = {
                    "round": attempt,
                    "status": status,
                    "solve_s_contended": elapsed,
                    "sample_count": len(samples),
                }
                diagnostics = solver.getResults()
                attempt_result["returned_nodes"] = solved.tolist()
                attempt_result["constraint_names"] = problem.getNLPConstraintNames()
                attempt_result["exact_constraint_violations"] = np.asarray(
                    problem.getExactConstraintViolations().raw
                ).tolist()
                attempt_result["approx_merit_improve"] = float(diagnostics.approx_merit_improve)
                attempt_result["exact_merit_improve"] = float(diagnostics.exact_merit_improve)
                attempt_result["merit_improve_ratio"] = float(diagnostics.merit_improve_ratio)
                attempt_result["overall_iteration"] = int(diagnostics.overall_iteration)
                attempt_result["penalty_iteration"] = int(diagnostics.penalty_iteration)
                sampled_velocity = model.evaluate(samples, solved, 1)[1]
                sample_violations = []
                for sample_index, sample_time in enumerate(samples):
                    for axis, (lower, upper) in enumerate(bounds[sample_index]):
                        value = float(sampled_velocity[sample_index, axis])
                        violation = max(lower - value, value - upper, 0.0)
                        if violation > 0.0:
                            sample_violations.append(
                                {
                                    "time_s": float(sample_time),
                                    "axis": names[axis],
                                    "value": value,
                                    "raw_violation": float(violation),
                                }
                            )
                attempt_result["worst_sampled_velocity_violations"] = sorted(
                    sample_violations, key=lambda item: item["raw_violation"], reverse=True
                )[:5]
                result["attempts"].append(attempt_result)
                if status != "SQPStatus.NLP_CONVERGED":
                    try:
                        failed_samples = np.linspace(0.0, stage_times[-1], SAMPLES)
                        failed_q = model.evaluate(failed_samples, solved, 0)[0]
                        attempt_result["failed_candidate_sampled_check"] = inspect_sampled_curve(
                            checker, failed_q, limits, SQP_FEASIBILITY_BAND
                        )
                    except Exception as check_exc:
                        attempt_result["failed_candidate_sampled_check_error"] = (
                            f"{type(check_exc).__name__}: {check_exc}"
                        )
                    raise FixtureConstrainedError(f"native SQP status {status}")
                actual = build_jerk_model(knots_s, v0, a0, v1, a1).curve(solved)
                position = position_extrema(actual)
                excess = [
                    max(limits[k, 0] - e["minimum"], e["maximum"] - limits[k, 1], 0.0)
                    for k, e in enumerate(position)
                ]
                if any(value > SQP_FEASIBILITY_BAND for value in excess):
                    raise FixtureConstrainedError(f"actual position bound excursion {excess}")
                violating_times = []
                extrema_rows = []
                for j in range(len(chain)):
                    extrema = velocity_extrema(actual, stage_times[j], stage_times[j + 1])
                    for axis, e in enumerate(extrema):
                        for kind in ("minimum", "maximum"):
                            t = e[kind + "_time_s"]
                            bound = fixture_velocity_bounds(stage_states, stage_times, [t], speeds)[
                                0
                            ][axis]
                            value = e[kind]
                            violation = max(bound[0] - value, value - bound[1], 0.0)
                            extrema_rows.append(
                                {
                                    "stage": chain[j]["stage"],
                                    "axis": names[axis],
                                    "kind": kind,
                                    "time_s": t,
                                    "value": value,
                                    "raw_violation": violation,
                                }
                            )
                            if violation > SQP_FEASIBILITY_BAND:
                                violating_times.append(t)
                attempt_result["maximum_raw_velocity_violation_native_units_s"] = max(
                    r["raw_violation"] for r in extrema_rows
                )
                attempt_result["violating_extrema_count"] = len(violating_times)
                attempt_result["velocity_extrema"] = extrema_rows
                if not violating_times:
                    seed = solved
                    break
                enlarged = np.unique(np.r_[samples, violating_times])
                if len(enlarged) == len(samples):
                    raise FixtureConstrainedError("actual derivative witness stagnated")
                samples = enlarged
                seed = solved
            else:
                raise FixtureConstrainedError("actual derivative witness budget exhausted")
            result["stages"] = []
            for j, stage in enumerate(chain):
                ts = np.linspace(stage_times[j], stage_times[j + 1], SAMPLES)
                q, velocity, _ = model.evaluate(ts, seed, 2)
                checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
                original_q = np.asarray(stage["states"])
                baseline = np.column_stack(
                    [
                        np.interp(
                            np.linspace(0, 1, SAMPLES),
                            np.linspace(0, 1, len(original_q)),
                            original_q[:, k],
                        )
                        for k in range(9)
                    ]
                )
                result["stages"].append(
                    {
                        "stage": stage["stage"],
                        "collision_and_position": checked,
                        "sampled_robot_speed_ratio": (
                            np.max(np.abs(velocity[:, :7]), axis=0) / np.asarray(speeds)
                        ).tolist(),
                        "max_departure_from_generated_chord_by_axis": np.max(
                            np.abs(q - baseline), axis=0
                        ).tolist(),
                        "target_error_by_axis": np.max(
                            np.abs(q[[0, -1]] - original_q[[0, -1]]), axis=0
                        ).tolist(),
                    }
                )
            result["status"] = "sampled_candidate_physical_and_process_acceptance_unresolved"
            result["solved_nodes"] = seed.tolist()
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        report["chains"].append(result)
        save()
    report["status"] = (
        "completed_with_failures"
        if any(r["status"] == "failed" for r in report["chains"])
        else "evaluated_unapproved_chains"
    )
    save()
    if snapshot(scene) != original:
        raise FixtureConstrainedError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "chains": [
                    (r["first_stage"], r["status"], r.get("error")) for r in report["chains"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
