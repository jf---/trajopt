"""Fresh graph planning followed by native time-aware transit refinement."""

from time import perf_counter

import _quintic_spline as native
import numpy as np
from continuous_spline_jerk import build_jerk_model
from fresh_joins import modeled_join_diagnostics
from fresh_prepare import prepare_fresh
from fresh_validation import CurveLimitError, inspect_sampled_curve
from native_transits_event_aware import refine_transits
from quintic_position_extrema import position_extrema

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Independent evaluation density, same as prior all-tour spline experiments.
SPLINE_SAMPLES = 1025
# Existing SQP constraint-feasibility band, in each axis's native m/rad units.
SQP_FEASIBILITY_BAND = 1e-4
# Binary64 allowance when a sampled ratio is at its declared speed bound.
SPEED_BOUND_ULPS = 32


class FreshTourError(RuntimeError):
    """The fresh tour or modeled motion is incomplete or invalid."""


class FreshSpeedError(RuntimeError):
    """The sampled modeled spline exceeds a declared robot speed."""


def modeled_tour_end_secants(segments, clock):
    """Return explicit generated-interpolation endpoint slopes, not physical data."""
    result = []
    for i, end in ((0, False), (-1, True)):
        stage = segments[i]
        timing = clock["stages"][i]
        q = np.asarray(stage["states"], dtype=float)
        duration = float(timing["duration_s"])
        if not stage["stage"].startswith("transit:") or duration <= 0:
            raise FreshTourError(
                "modeled endpoint secant requires positive-duration endpoint transit"
            )
        result.append(((q[-1] - q[-2]) if end else (q[1] - q[0])) * (len(q) - 1) / duration)
    return tuple(result)


def refine_checked_tour(
    scene, tour, record, checked, raw, metadata, *, boundary_policy, transit_solver=None
):
    """Return checked native transit curves with unchanged process and explicit joins.

    The caller supplies a boundary policy returning nine-axis velocities in
    native units per second. The output is finite-sampled, not a physical timing,
    continuous-clearance or whole-tour jerk-continuity claim.
    """
    env, program, source, tours, names, _, _, _, _, home, info, _ = scene["inputs"]
    start = perf_counter()
    segments, clock, speeds = prepare_fresh(
        env, program, source, tours, home, names, raw, metadata, tour, record, checked
    )
    start_velocity, end_velocity = boundary_policy(segments, clock)
    preparation_s = perf_counter() - start
    start = perf_counter()
    solver = refine_transits if transit_solver is None else transit_solver
    outcomes = solver(
        env,
        info,
        names,
        segments,
        clock,
        speeds,
        start_velocity=start_velocity,
        end_velocity=end_velocity,
    )
    refine_s = perf_counter() - start
    start = perf_counter()
    limits = np.asarray(
        env.getJointGroup("fresh_native_full_limits", list(names)).getLimits().joint_limits,
        dtype=float,
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    endpoint_states = {}
    for row in outcomes:
        v0, v1 = np.asarray(row["v0"]), np.asarray(row["v1"])
        if transit_solver is None:
            # The original refiner explicitly constructs zero endpoint accelerations.
            a0, a1 = np.zeros(7), np.zeros(7)
        else:
            a0, a1 = np.asarray(row["a0"]), np.asarray(row["a1"])
        model = native.QuinticSplineModel.build(row["knots_s"], v0, a0, v1, a1)
        nodes = np.asarray(row["nodes"])
        times = np.linspace(0, row["duration_s"], SPLINE_SAMPLES)
        pos, vel = model.evaluate(times, nodes, 1)
        fixture = np.asarray(row["fixture"])
        full = np.column_stack((pos, np.tile(fixture, (len(pos), 1))))
        expected = np.asarray(
            next(stage["states"] for stage in segments if stage["stage"] == row["stage"])
        )
        endpoint_error = np.max(np.abs(full[[0, -1]] - expected[[0, -1]]), axis=0)
        row["endpoint_raw_residual_by_axis"] = endpoint_error.tolist()
        if np.any(endpoint_error > SQP_FEASIBILITY_BAND):
            raise FreshTourError(
                f"{row['stage']} endpoint position differs: {endpoint_error.tolist()}"
            )
        row["actual_curve_check"] = inspect_sampled_curve(
            checker, full, limits, SQP_FEASIBILITY_BAND
        )
        peak = np.max(np.abs(vel), axis=0) / np.asarray(speeds)
        row["sampled_speed_ratio_by_axis"] = peak.tolist()
        if np.any(peak > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps):
            raise FreshSpeedError(
                f"{row['stage']} sampled speed ratio exceeds one: {peak.tolist()}"
            )
        curve = build_jerk_model(row["knots_s"], v0, a0, v1, a1).curve(nodes)
        extrema = position_extrema(curve)
        raw_excursions = [
            max(limits[i, 0] - item["minimum"], item["maximum"] - limits[i, 1], 0.0)
            for i, item in enumerate(extrema)
        ]
        row["position_extrema"] = extrema
        row["extrema_raw_limit_excursions"] = raw_excursions
        if any(x > SQP_FEASIBILITY_BAND for x in raw_excursions):
            raise CurveLimitError(
                f"{row['stage']} position extrema exceed SQP feasibility band: {raw_excursions}"
            )
        endpoint_states[row["stage"]] = {
            "start": (full[0], np.r_[vel[0], np.zeros(2)]),
            "end": (full[-1], np.r_[vel[-1], np.zeros(2)]),
        }
    joins = modeled_join_diagnostics(segments, clock, endpoint_states)
    checking_s = perf_counter() - start
    return {
        "status": "sampled_process_and_actual_transits_with_unresolved_join_diagnostics",
        "scope": "Fresh generated source; unchanged sampled process stages, native fixed-clock transit splines; finite collision samples, numerical position extrema, modeled joins, no physical or full-tour smoothness claim.",
        "tour": tour,
        "preparation_s": preparation_s,
        "refinement_s": refine_s,
        "checking_s": checking_s,
        "clock": clock,
        "generated_stages": segments,
        "transits": outcomes,
        "join_diagnostics": joins,
        "boundary_velocity_source": boundary_policy.__name__,
        "process_verification": {k: v for k, v in checked.items() if k != "expanded"},
    }


def plan_and_refine(scene, tour, plan, verify, raw, metadata, *, boundary_policy):
    """Generate a fresh tour and refine it in the caller's already prepared scene."""
    start = perf_counter()
    record = plan(
        *scene["inputs"],
        tour=tour,
        use_cached_graph=True,
        reuse_probe_rungs=True,
        graph_budget_s=120.0,
        captured_frontier=None,
        route_transits=True,
        transit_backend="ompl_piqp",
        route_fixtures=True,
        fixture_backend="home_park_search",
        reuse_home_sweeps=True,
        fixture_home_only=True,
        feasibility_first=True,
        defer_unresolved_edges=True,
        validate_weld_intervals=True,
    )
    planning_s = perf_counter() - start
    if record.get("status") != "sampled_nodes_only":
        raise FreshTourError(f"tour {tour} fresh plan status: {record.get('status')}")
    start = perf_counter()
    checked = verify(*scene["verify_inputs"], record, sample_all_intervals=True)
    verification_s = perf_counter() - start
    result = refine_checked_tour(
        scene,
        tour,
        record,
        checked,
        raw,
        metadata,
        boundary_policy=boundary_policy,
    )
    result["planning_s"] = planning_s
    result["initial_verification_s"] = verification_s
    return result
