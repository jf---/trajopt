"""Opt-in fixed-chord fixture easing and actual native transit checks."""

from copy import deepcopy
from functools import partial
from time import perf_counter

import _quintic_spline as native
import numpy as np
from continuous_spline_jerk import build_jerk_model
from eased_boundaries import eased_transit_boundaries
from fixture_chain_clock import FixtureClockError, evaluate_fixture_clock, fixture_chain_clock
from fresh_joins import modeled_join_diagnostics
from fresh_tour_pipeline import (
    SPEED_BOUND_ULPS,
    SPLINE_SAMPLES,
    SQP_FEASIBILITY_BAND,
    FreshSpeedError,
    FreshTourError,
    modeled_tour_end_secants,
)
from fresh_validation import CurveLimitError, inspect_sampled_curve
from native_transits_event_aware import refine_transits
from quintic_position_extrema import position_extrema

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters


def run_fixture_chain_trial(env, info, names, seg, clk, sp):
    """Refine supplied checked stages using one eased fixture-chain clock.

    The caller must independently establish process geometry and timing provenance.
    Zero or multiple disjoint fixture chains, collision failures and incompatible
    fixed-fixture joins raise; no fallback or physical timing claim is made.
    """
    # The caller supplies freshly generated and independently verified stages.
    start = perf_counter()
    newclock = deepcopy(clk)
    fixture_indices = [i for i, x in enumerate(seg) if x["stage"].startswith("fixture:")]
    if not fixture_indices or fixture_indices != list(
        range(fixture_indices[0], fixture_indices[-1] + 1)
    ):
        raise FixtureClockError("expected one contiguous fixture chain")
    chain = [seg[i] for i in fixture_indices]
    old_times = [0.0]
    for i in fixture_indices:
        old_times.append(old_times[-1] + float(clk["stages"][i]["duration_s"]))
    fixture_model = fixture_chain_clock(chain, old_times[-1])
    endpoint_velocities = {}
    endpoint_actual = {}
    for k, i in enumerate(fixture_indices):
        label = seg[i]["stage"]
        t0, t1 = fixture_model["stage_times_s"][k : k + 2]
        q0, v0, _ = evaluate_fixture_clock(fixture_model, t0)
        q1, v1, _ = evaluate_fixture_clock(fixture_model, t1)
        endpoint_velocities[label] = (v0, v1)
        endpoint_actual[label] = {"start": (q0, v0), "end": (q1, v1)}
        newclock["stages"][i]["duration_s"] = float(t1 - t0)
    # Retiming preserves total chain duration; each individual event retains order.
    newclock["node_times_s"] = [0.0]
    for stage, timing in zip(seg, newclock["stages"], strict=True):
        t0 = newclock["node_times_s"][-1]
        duration = float(timing["duration_s"])
        if duration:
            newclock["node_times_s"].extend(
                float(t0 + duration * j / (len(stage["states"]) - 1))
                for j in range(1, len(stage["states"]))
            )
    start_velocity, end_velocity = modeled_tour_end_secants(seg, newclock)
    resolver = partial(eased_transit_boundaries, fixture_endpoint_velocities=endpoint_velocities)
    outcomes = refine_transits(
        env,
        info,
        names,
        seg,
        newclock,
        sp,
        start_velocity=start_velocity,
        end_velocity=end_velocity,
        boundary_resolver=resolver,
    )
    solve_elapsed = perf_counter() - start
    limits = np.asarray(
        env.getJointGroup("fresh_native_fixture_full", list(names)).getLimits().joint_limits,
        dtype=float,
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    fixture_times = np.linspace(0, fixture_model["duration_s"], SPLINE_SAMPLES)
    fixture_samples = np.asarray(
        [evaluate_fixture_clock(fixture_model, t)[0] for t in fixture_times]
    )
    fixture_check = inspect_sampled_curve(checker, fixture_samples, limits, SQP_FEASIBILITY_BAND)
    for row in outcomes:
        v0, v1 = np.asarray(row["v0"]), np.asarray(row["v1"])
        zero = np.zeros(7)
        nodes = np.asarray(row["nodes"])
        model = native.QuinticSplineModel.build(row["knots_s"], v0, zero, v1, zero)
        pos, vel = model.evaluate(np.linspace(0, row["duration_s"], SPLINE_SAMPLES), nodes, 1)
        full = np.column_stack((pos, np.tile(row["fixture"], (len(pos), 1))))
        expected = np.asarray(next(s["states"] for s in seg if s["stage"] == row["stage"]))
        row["endpoint_max_raw_residual"] = float(np.max(np.abs(full[[0, -1]] - expected[[0, -1]])))
        if row["endpoint_max_raw_residual"] > SQP_FEASIBILITY_BAND:
            raise FreshTourError(f"{row['stage']} endpoint residual")
        row["actual_curve_check"] = inspect_sampled_curve(
            checker, full, limits, SQP_FEASIBILITY_BAND
        )
        row["sampled_speed_ratio_by_axis"] = (np.max(np.abs(vel), axis=0) / np.asarray(sp)).tolist()
        if max(row["sampled_speed_ratio_by_axis"]) > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps:
            raise FreshSpeedError(f"{row['stage']} overspeed")
        curve = build_jerk_model(row["knots_s"], v0, zero, v1, zero).curve(nodes)
        extrema = position_extrema(curve)
        row["extrema"] = extrema
        row["extrema_raw_excess"] = [
            float(max(limits[i, 0] - x["minimum"], x["maximum"] - limits[i, 1], 0))
            for i, x in enumerate(extrema)
        ]
        if max(row["extrema_raw_excess"]) > SQP_FEASIBILITY_BAND:
            raise CurveLimitError(f"{row['stage']} extrema bound")
        endpoint_actual[row["stage"]] = {
            "start": (full[0], np.r_[vel[0], np.zeros(2)]),
            "end": (full[-1], np.r_[vel[-1], np.zeros(2)]),
        }
    joins = modeled_join_diagnostics(seg, newclock, endpoint_actual)
    result = {
        "scope": "Opt-in fixed nine-axis fixture-chord quintic easing and fresh native transit splines; finite sampled checks, modeled clock, no fixture speed certification or physical timing.",
        "original_fixture_stage_times_s": old_times,
        "fixture": fixture_model,
        "fixture_check": fixture_check,
        "clock": newclock,
        "transits": outcomes,
        "join_diagnostics": joins,
        "solve_elapsed_s": solve_elapsed,
        "total_elapsed_s": perf_counter() - start,
    }
    return result
