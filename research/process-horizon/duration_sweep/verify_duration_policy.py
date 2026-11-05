"""Independently sample one duration policy without changing the retained scene."""

import json
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import (
    check_stage_samples,
    fixture_stage,
    process_stage,
    transit_stage,
    weld_stage,
)
from duration_policy import scale_transit_clock, validate_fixture_policy, validate_transit_results
from free_space_transit import _restore_scene
from fresh_scene import snapshot
from process_geometry import assess_process_geometry
from scipy.interpolate import CubicSpline

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Fixed sample budgets for transit, process, and local blend windows.
TRANSIT_SAMPLES = 1025
PROCESS_SAMPLES = 257
LOCAL_WINDOW_SAMPLES = 65


class TransitDurationError(RuntimeError):
    """A transit-clock override is unknown, nonfinite or nonpositive."""


def apply_transit_durations(clock, overrides):
    """Return a copied modeled clock with only selected transit durations changed."""
    names = {item["stage"] for item in clock["stages"] if item["stage"].startswith("transit:")}
    if set(overrides) - names:
        raise TransitDurationError("duration overrides must name existing transits")
    result = []
    for stage in clock["stages"]:
        item = dict(stage)
        if item["stage"] in overrides:
            value = float(overrides[item["stage"]])
            if not np.isfinite(value) or value <= 0:
                raise TransitDurationError("transit duration must be finite and positive")
            item["duration_s"] = value
        result.append(item)
    return {**clock, "stages": result}


class ProcessRateError(RuntimeError):
    """The process rate override would alter a probe corner or is invalid."""


def process_rates_for_stage(label, overrides):
    """Return normalized endpoint progress rates, preserving probe corners."""
    phase = label.partition(":")[0]
    if phase not in ("approach", "search", "retract", "withdraw"):
        raise ProcessRateError("process phase required")
    default = (1.0 if phase == "approach" else 0.0, 1.0 if phase == "withdraw" else 0.0)
    if label not in overrides:
        return default
    values = np.asarray(overrides[label], dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ProcessRateError("finite nonnegative endpoint rates required")
    if (phase != "approach" and values[0] != 0) or (phase != "withdraw" and values[1] != 0):
        raise ProcessRateError("probe process corners must retain their zero-rate boundary")
    return float(values[0]), float(values[1])


def verify_duration_policy(
    duration_factor,
    scene,
    segments,
    clock,
    speed,
    output,
    artifacts,
    fixture_policy,
    *,
    process_rates=None,
    transit_durations=None,
    stage_sink=None,
):
    """Check all actual curves for one explicit transit-duration policy.

    The returned result retains modeled stage times, position/velocity/
    acceleration joins, sampled collision and limits; it is never physical
    or manufacturing acceptance.
    """
    env, _, _, _, names, _, _, _, _, _, _, _ = scene["inputs"]
    process_rates = {} if process_rates is None else process_rates
    valid_process_labels = {
        s["stage"]
        for s in segments
        if s["stage"].split(":")[0] in ("approach", "search", "retract", "withdraw")
    }
    if set(process_rates) - valid_process_labels:
        raise ProcessRateError("unknown process rate override stage")
    before = snapshot(scene)
    factor = float(duration_factor)
    policy_dir = output / f"global-duration-{factor:g}"
    policy_summary_path = policy_dir / "summary.json"
    transit_summary = json.loads(policy_summary_path.read_text())
    if float(transit_summary["factor"]) != factor:
        raise RuntimeError("duration policy factor does not match summary")
    policy = json.loads(fixture_policy.read_text())["fixture"]
    validate_fixture_policy(segments, policy)
    policy_clock_data = scale_transit_clock(
        stages=segments, clock=clock, factor=factor, fixture_times=policy["stage_times_s"]
    )
    if transit_durations is not None:
        policy_clock_data = apply_transit_durations(policy_clock_data, transit_durations)
    transits = validate_transit_results(segments, policy_clock_data, transit_summary["transits"])
    if len(segments) != 67 or len(transits) != 14:
        raise RuntimeError("expected exactly 67 stages and 14 unique Tour09 transits")
    clock_stages = clock["stages"]
    probe_dir = artifacts / "all-probes-pass1"
    probe_data = {
        i: json.loads((probe_dir / f"probe{i:02d}" / "ik-progress-report.json").read_text())
        for i in range(9)
    }
    probe_curves = {
        i: [
            CubicSpline(np.linspace(0, 1, len(x)), np.asarray(x), axis=0)
            for x in probe_data[i]["stage_states"]
        ]
        for i in range(9)
    }
    local = json.loads((artifacts / "weld-local-extent-1.json").read_text())
    local_curve = CubicSpline(
        np.asarray(local["joint_times_s"]),
        np.asarray(local["joint_states"]),
        axis=0,
        bc_type="natural",
    )
    local_labels = ["weld:4", "weld:5", "weld:6", "weld:7", "weld:8", "weld:9"]
    local_durations = np.asarray(
        [
            float(
                clock["stages"][next(j for j, x in enumerate(segments) if x["stage"] == name)][
                    "duration_s"
                ]
            )
            for name in local_labels
        ]
    )
    local_edges = np.r_[0, np.cumsum(local_durations)]
    local_map = {name: (k, local, local_curve, local_edges) for k, name in enumerate(local_labels)}
    for first, lab in (
        (20, ["weld:20", "weld:21", "weld:22"]),
        (27, ["weld:27", "weld:28", "weld:29"]),
    ):
        data = json.loads((artifacts / f"local-weld-{first}/weld-local-extent-1.json").read_text())
        curve = CubicSpline(
            np.asarray(data["joint_times_s"]),
            np.asarray(data["joint_states"]),
            axis=0,
            bc_type="natural",
        )
        ds = np.asarray(
            [
                float(
                    clock["stages"][next(j for j, x in enumerate(segments) if x["stage"] == name)][
                        "duration_s"
                    ]
                )
                for name in lab
            ]
        )
        e = np.r_[0, np.cumsum(ds)]
        local_map.update({name: (k, data, curve, e) for k, name in enumerate(lab)})
    fixture_indices = [i for i, s in enumerate(segments) if s["stage"].startswith("fixture:")]
    fixture_start = np.asarray(policy["start"])
    fixture_end = np.asarray(policy["end"])
    fixture_times = np.asarray(policy["stage_times_s"])
    fixture_duration = float(policy["duration_s"])
    expected = len(segments)
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("tour09-assembly-nine", list(names)).getLimits().joint_limits
    )
    if len(fixture_indices) + 1 != len(fixture_times) or np.any(np.diff(fixture_times) <= 0):
        raise RuntimeError("fixture timing has invalid event layout")
    policy_clock = []
    current_time = 0.0
    for index, stage in enumerate(segments):
        label = stage["stage"]
        if label in transits:
            duration = float(transits[label]["duration_s"])
        elif index in fixture_indices:
            fixture_index = fixture_indices.index(index)
            duration = float(fixture_times[fixture_index + 1] - fixture_times[fixture_index])
        else:
            duration = float(clock_stages[index]["duration_s"])
        if not np.isfinite(duration) or duration < 0:
            raise RuntimeError(f"{label}: invalid stage duration")
        policy_clock.append(
            {
                "stage": label,
                "start_s": current_time,
                "end_s": current_time + duration,
                "duration_s": duration,
            }
        )
        current_time += duration
    initial = {str(k): float(v) for k, v in env.getState().joints.items()}
    revision = int(env.getRevision())
    started = perf_counter()
    stages = []
    all_hits = []
    all_excess = np.zeros(9)
    try:
        for idx, stage in enumerate(segments):
            label = stage["stage"]
            phase, _, number = label.partition(":")
            source = np.asarray(stage["states"])
            fixture = source[0, 7:]
            duration = float(policy_clock[idx]["duration_s"])
            undefined_internal_velocities = False
            if phase == "transit":
                item = transits[label]
                duration = float(item["duration_s"])
                model = native.QuinticSplineModel.build(
                    item["knots_s"],
                    np.asarray(item["v0"]),
                    np.asarray(item["a0"]),
                    np.asarray(item["v1"]),
                    np.asarray(item["a1"]),
                )
                ts = np.linspace(0, item["duration_s"], TRANSIT_SAMPLES)
                position, velocity, acceleration = model.evaluate(ts, np.asarray(item["nodes"]), 2)
                jerk_map = model.map(ts.tolist(), 3)
                jerk = jerk_map.basis @ np.asarray(item["nodes"]) + jerk_map.offset
                sample_times = ts
                q = np.column_stack((position, np.tile(fixture, (len(position), 1))))
                v = np.column_stack((velocity, np.zeros((len(velocity), 2))))
                a = np.column_stack((acceleration, np.zeros((len(acceleration), 2))))
                if stage_sink is not None:
                    actual_stage = transit_stage(label, duration, model, item["nodes"], fixture)
            elif phase in ("approach", "search", "retract", "withdraw"):
                n = int(number)
                leg = ["approach", "search", "retract", "withdraw"].index(phase)
                curve = probe_curves[n][leg]
                duration = float(clock["stages"][idx]["duration_s"])
                tau = np.linspace(0, 1, PROCESS_SAMPLES)
                v0, v1 = process_rates_for_stage(label, process_rates)
                A = 10 - 6 * v0 - 4 * v1
                B = -15 + 8 * v0 + 7 * v1
                C = 6 - 3 * v0 - 3 * v1
                s = v0 * tau + A * tau**3 + B * tau**4 + C * tau**5
                sd = (v0 + 3 * A * tau**2 + 4 * B * tau**3 + 5 * C * tau**4) / duration
                sdd = (6 * A * tau + 12 * B * tau**2 + 20 * C * tau**3) / duration**2
                sddd = (6 * A + 24 * B * tau + 60 * C * tau**2) / duration**3
                s = np.clip(s, 0, 1)
                position = curve(s)
                velocity = curve(s, 1) * sd[:, None]
                acceleration = curve(s, 2) * sd[:, None] ** 2 + curve(s, 1) * sdd[:, None]
                jerk = (
                    curve(s, 3) * sd[:, None] ** 3
                    + 3 * curve(s, 2) * sd[:, None] * sdd[:, None]
                    + curve(s, 1) * sddd[:, None]
                )
                sample_times = tau * duration
                q = np.column_stack((position, np.tile(fixture, (len(position), 1))))
                v = np.column_stack((velocity, np.zeros((len(velocity), 2))))
                a = np.column_stack((acceleration, np.zeros((len(acceleration), 2))))
                if stage_sink is not None:
                    actual_stage = process_stage(label, duration, curve, fixture, v0, v1)
            elif phase == "fixture":
                k = fixture_indices.index(idx)
                ts = np.linspace(fixture_times[k], fixture_times[k + 1], PROCESS_SAMPLES)
                duration = float(fixture_times[k + 1] - fixture_times[k])
                u = ts / fixture_duration
                s = 10 * u**3 - 15 * u**4 + 6 * u**5
                sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / fixture_duration
                sdd = (60 * u - 180 * u**2 + 120 * u**3) / fixture_duration**2
                sddd = (60 - 360 * u + 360 * u**2) / fixture_duration**3
                sample_times = ts
                delta = fixture_end - fixture_start
                q = fixture_start + delta[None, :] * s[:, None]
                v = delta[None, :] * sd[:, None]
                a = delta[None, :] * sdd[:, None]
                jerk = delta[None, :] * sddd[:, None]
                if stage_sink is not None:
                    actual_stage = fixture_stage(
                        label, policy, fixture_times[k], fixture_times[k + 1]
                    )
            elif phase == "weld" and label in local_map:
                k, local, local_curve, local_edges = local_map[label]
                ts = np.linspace(local_edges[k], local_edges[k + 1], PROCESS_SAMPLES).tolist()
                left = local["left_blend_windows_s"][k]
                right = local["right_blend_windows_s"][k]
                if left > 0:
                    ts.extend(
                        np.linspace(
                            local_edges[k], local_edges[k] + left, LOCAL_WINDOW_SAMPLES
                        ).tolist()
                    )
                if right > 0:
                    ts.extend(
                        np.linspace(
                            local_edges[k + 1] - right, local_edges[k + 1], LOCAL_WINDOW_SAMPLES
                        ).tolist()
                    )
                ts = np.unique(np.asarray(ts))
                ts = ts[
                    np.r_[
                        True,
                        np.diff(ts)
                        > 16
                        * np.finfo(float).eps
                        * np.maximum(1.0, np.maximum(np.abs(ts[:-1]), np.abs(ts[1:]))),
                    ]
                ]
                position = local_curve(ts)
                velocity = local_curve(ts, 1)
                acceleration = local_curve(ts, 2)
                jerk = local_curve(ts, 3)
                sample_times = ts
                q = np.column_stack((position, np.tile(fixture, (len(position), 1))))
                v = np.column_stack((velocity, np.zeros((len(velocity), 2))))
                a = np.column_stack((acceleration, np.zeros((len(acceleration), 2))))
                if stage_sink is not None:
                    actual_stage = weld_stage(label, duration, local_curve, local_edges[k], fixture)
            elif phase == "weld":
                raise RuntimeError(f"{label}: missing explicit local weld blend")
            else:
                raise RuntimeError(f"unsupported stage {label}")
            if stage_sink is not None:
                check_stage_samples(actual_stage, sample_times, q, v, a)
                stage_sink(actual_stage)
            if jerk.shape[1] == 7:
                jerk = np.column_stack((jerk, np.zeros((len(jerk), 2))))
            if q.shape[1] != 9 or not np.isfinite(q).all() or not np.isfinite(jerk).all():
                raise RuntimeError(f"invalid stage {label}")
            process_geometry = None
            if phase in ("approach", "search", "retract", "withdraw", "weld"):

                def workpiece_pose(state):
                    current = env.getState(dict(zip(names, map(float, state), strict=True)))
                    part = np.asarray(current.link_transforms["workpiece_link"].matrix)
                    tcp = np.asarray(current.link_transforms["nozzle_link"].matrix)
                    return np.linalg.solve(part, tcp)

                first = workpiece_pose(source[0])
                last = workpiece_pose(source[-1])
                actual = np.asarray([workpiece_pose(row) for row in q])
                process_geometry = assess_process_geometry(first, last, actual)
            all_excess = np.maximum(
                all_excess,
                np.max(np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0), axis=0),
            )
            hits = []
            for n, state in enumerate(q):
                for h in checker.contacts(state):
                    if float(h.distance_m) < 0:
                        hits.append(
                            {
                                "stage": label,
                                "sample": n,
                                "distance_m": float(h.distance_m),
                                "links": [str(x) for x in h.links],
                            }
                        )
            all_hits.extend(hits)
            stages.append(
                {
                    "stage": label,
                    "duration_s": duration,
                    "samples": len(q),
                    "source_endpoint_start_gap": float(np.max(np.abs(q[0] - source[0]))),
                    "source_endpoint_end_gap": float(np.max(np.abs(q[-1] - source[-1]))),
                    "first_q": q[0].tolist(),
                    "last_q": q[-1].tolist(),
                    "first_qd": v[0].tolist(),
                    "last_qd": v[-1].tolist(),
                    "first_qdd": a[0].tolist(),
                    "last_qdd": a[-1].tolist(),
                    "max_sourced_arm_track_speed_ratio": float(
                        np.max(np.abs(v[:, :7]) / np.asarray(speed))
                    ),
                    "negative_contacts": len(hits),
                    "max_velocity_by_axis": np.max(np.abs(v), axis=0).tolist(),
                    "max_acceleration_by_axis": np.max(np.abs(a), axis=0).tolist(),
                    "max_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                    "integrated_squared_jerk_estimate_by_axis": np.trapezoid(
                        jerk * jerk, sample_times, axis=0
                    ).tolist(),
                    "process_geometry": process_geometry,
                    "weld_piecewise_linear_derivatives_discontinuous": undefined_internal_velocities,
                }
            )
        joins = []
        for a, b in zip(stages[:-1], stages[1:], strict=True):
            dq = np.asarray(b["first_q"]) - np.asarray(a["last_q"])
            dv = np.asarray(b["first_qd"]) - np.asarray(a["last_qd"])
            da = np.asarray(b["first_qdd"]) - np.asarray(a["last_qdd"])
            joins.append(
                {
                    "left": a["stage"],
                    "right": b["stage"],
                    "max_position_gap": float(np.max(np.abs(dq))),
                    "max_velocity_jump": float(np.max(np.abs(dv))),
                    "max_acceleration_jump": float(np.max(np.abs(da))),
                    "velocity_jump_by_axis": dv.tolist(),
                    "acceleration_jump_by_axis": da.tolist(),
                    "weld_linear_knot_derivatives_may_be_undefined": bool(
                        a["weld_piecewise_linear_derivatives_discontinuous"]
                        or b["weld_piecewise_linear_derivatives_discontinuous"]
                    ),
                }
            )
        result = {
            "scope": "Generated Tour09 experimental assembly: 9 IK-derived probe curves, 14 native refined transits, 5 eased fixture stages, three generated local weld blends covering all 12 lead/weld stages. Controller blending remains unmodeled and cubic jerk can jump at interpolation knots. Modeled timing, finite sampled checks only; overall acceptance is not established.",
            "status": "sampled_assembly_only",
            "duration_factor": factor,
            "transit_policy_summary": str(policy_summary_path),
            "transit_policy_elapsed_s": transit_summary.get("elapsed_s"),
            "modeled_tour_duration_s": current_time,
            "modeled_stage_clock": policy_clock,
            "no_hit_assumption": "Each SearchL endpoint is a hypothetical commanded limit, not an observed hit.",
            "stage_count": len(stages),
            "checked_samples": sum(x["samples"] for x in stages),
            "elapsed_s": perf_counter() - started,
            "stages": stages,
            "joins": joins,
            "negative_contacts": len(all_hits),
            "worst_contact": min(all_hits, key=lambda x: x["distance_m"]) if all_hits else None,
            "raw_limit_excess": all_excess.tolist(),
            "overall_accepted": False,
            "jerk_integration": "sampled trapezoid estimate; cubic jerk may jump at knots",
        }
        if len(stages) != expected:
            raise RuntimeError("incomplete stage coverage")
        (policy_dir / "tour09-duration-policy-assembled.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(
            "TOUR09_DURATION_ASSEMBLY",
            json.dumps({k: v for k, v in result.items() if k not in ("stages", "joins")}),
            flush=True,
        )
    finally:
        _restore_scene(env, initial, revision)
    if snapshot(scene) != before:
        raise RuntimeError("scene changed")
    return result
