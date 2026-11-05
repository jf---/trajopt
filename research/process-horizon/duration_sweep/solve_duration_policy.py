"""Run one uniform transit-clock policy with unchanged process derivatives."""

import json
from time import perf_counter

import _quintic_spline as native
import numpy as np
from all24_spline_scene import native_log
from duration_policy import scale_transit_clock, validate_fixture_policy, validate_transit_results
from fresh_scene import snapshot
from fresh_validation import inspect_sampled_curve
from native_accel_transits import refine_transits

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Numerical source-state equality screen, not a process tolerance.
STATE_ROUNDOFF_ULPS = 128
# Existing SQP feasibility setting in native joint units, not a physical tolerance.
SQP_FEASIBILITY_BAND = 1e-4
TRANSIT_CHECK_SAMPLES = 1025


class TransitSweepError(RuntimeError):
    """A stage, boundary or solver result invalidates this duration policy."""


def solve_duration_policy(factor, scene, segments, clock, speed, output, artifacts, fixture_policy):
    """Replan every transit once and independently check the actual spline."""
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    before = snapshot(scene)
    fixture = json.loads(fixture_policy.read_text())["fixture"]
    validate_fixture_policy(segments, fixture)
    policy_clock = scale_transit_clock(segments, clock, factor, fixture["stage_times_s"])
    baseline = json.loads((artifacts / "tour09-all-weld-blends-assembled.json").read_text())
    if [x["stage"] for x in baseline["stages"]] != [x["stage"] for x in segments]:
        raise TransitSweepError("generated stage sequence differs from the baseline assembly")
    if baseline["negative_contacts"] != 0:
        raise TransitSweepError("baseline assembly has sampled contacts")
    if len(segments) != 67 or sum(s["stage"].startswith("transit:") for s in segments) != 14:
        raise TransitSweepError("expected all 67 Tour09 stages and 14 transits")
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("duration-sweep-nine", list(names)).getLimits().joint_limits
    )
    folder = output / f"global-duration-{factor:g}"
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    started = perf_counter()
    for index, stage in enumerate(segments):
        if not stage["stage"].startswith("transit:"):
            continue
        stage_started = perf_counter()
        reference = baseline["stages"][index]
        first = np.asarray(reference["first_q"], dtype=float)
        last = np.asarray(reference["last_q"], dtype=float)
        original = np.asarray(stage["states"], dtype=float)
        band = STATE_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, float(np.max(np.abs(original))))
        if np.max(np.abs(first - original[0])) > band or np.max(np.abs(last - original[-1])) > band:
            raise TransitSweepError(f"{stage['stage']}: source and baseline endpoints differ")
        left = np.asarray(reference["first_qd"], dtype=float)
        right = np.asarray(reference["last_qd"], dtype=float)
        a0 = np.asarray(reference["first_qdd"], dtype=float)
        a1 = np.asarray(reference["last_qdd"], dtype=float)
        if any(x.shape != (9,) or not np.isfinite(x).all() for x in (left, right, a0, a1)):
            raise TransitSweepError(f"{stage['stage']}: invalid modeled boundary derivatives")
        if any(np.any(x[7:] != 0) for x in (left, right, a0, a1)):
            raise TransitSweepError(f"{stage['stage']}: fixture boundary derivative is nonzero")
        selected = [
            {
                "stage": item["stage"]
                if j == index or not item["stage"].startswith("transit:")
                else "other-" + item["stage"],
                "states": item["states"],
            }
            for j, item in enumerate(segments)
        ]
        selected_clock = [
            {**timing, "stage": selected[j]["stage"]}
            for j, timing in enumerate(policy_clock["stages"])
        ]

        def resolver(*_args, **_kwargs):
            return left, right, ()

        def accelerations(_index):
            return a0, a1

        curve = None
        try:
            with native_log(folder / f"{stage['stage'].replace(':', '-')}.log"):
                outcomes = refine_transits(
                    env,
                    info,
                    names,
                    selected,
                    {"stages": selected_clock},
                    speed,
                    start_velocity=left,
                    end_velocity=right,
                    boundary_resolver=resolver,
                    acceleration_resolver=accelerations,
                )
            if len(outcomes) != 1:
                raise TransitSweepError(f"{stage['stage']}: expected one solved transit")
            curve = outcomes[0]
            model = native.QuinticSplineModel.build(
                curve["knots_s"],
                np.asarray(curve["v0"]),
                np.asarray(curve["a0"]),
                np.asarray(curve["v1"]),
                np.asarray(curve["a1"]),
            )
            times = np.linspace(0, curve["duration_s"], TRANSIT_CHECK_SAMPLES)
            nodes = np.asarray(curve["nodes"])
            positions, velocities, accelerations = model.evaluate(times, nodes, 2)
            mapping = model.map(times.tolist(), 3)
            jerk = mapping.basis @ nodes + mapping.offset
            full = np.column_stack(
                (positions, np.tile(np.asarray(curve["fixture"]), (len(positions), 1)))
            )
            checked = inspect_sampled_curve(checker, full, limits, SQP_FEASIBILITY_BAND)
            quadrature = model.jerk_quadrature()
            values = quadrature.map.basis @ nodes + quadrature.map.offset
            integrated = np.sum(np.asarray(quadrature.weights)[:, None] * values * values, axis=0)
            result = {
                "stage": stage["stage"],
                "status": "sampled_validated",
                "wall_s": perf_counter() - stage_started,
                "solver_s": curve["solver_s"],
                "checks": checked,
                "curve": curve,
                "max_velocity_by_axis": np.max(np.abs(velocities), axis=0).tolist(),
                "max_acceleration_by_axis": np.max(np.abs(accelerations), axis=0).tolist(),
                "max_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                "integrated_squared_jerk_by_axis": integrated.tolist(),
                "max_sourced_speed_ratio": float(np.max(np.abs(velocities) / np.asarray(speed))),
                "left_position_gap": float(np.max(np.abs(full[0] - first))),
                "right_position_gap": float(np.max(np.abs(full[-1] - last))),
                "left_velocity_gap": float(np.max(np.abs(velocities[0] - left[:7]))),
                "right_velocity_gap": float(np.max(np.abs(velocities[-1] - right[:7]))),
                "left_acceleration_gap": float(np.max(np.abs(accelerations[0] - a0[:7]))),
                "right_acceleration_gap": float(np.max(np.abs(accelerations[-1] - a1[:7]))),
            }
        except Exception as exc:
            failure = {
                "stage": stage["stage"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "curve": curve,
            }
            (folder / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise TransitSweepError(
                f"{stage['stage']}: transit solve or verification failed"
            ) from exc
        results.append(result)
        summary = {"factor": factor, "elapsed_s": perf_counter() - started, "transits": results}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    validate_transit_results(segments, policy_clock, results)
    if snapshot(scene) != before:
        raise TransitSweepError("scene changed during duration policy")
    return summary
