"""Evaluate generated fixture chords with native ordered fixture-axis motion."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from continuous_spline_jerk import build_jerk_model
from fixture_multichord import build_fixture_multichord, fixture_knot_times
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries
from spline_velocity_extrema import velocity_extrema

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent finite collision sampling policy per stage.
PARITY_ULPS = 32768  # Binary64 evaluation parity, not a physical tolerance.


class FixtureMonotoneTrialError(RuntimeError):
    """Generated fixture data or actual motion checks are invalid."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "recovery", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chain", action="append", required=True)
    parser.add_argument("--duration-factor", action="append", type=float, required=True)
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.duration_factor):
        raise FixtureMonotoneTrialError("positive modeled duration factors required")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise FixtureMonotoneTrialError("verified generated recovery required")
    if args.output.exists():
        raise FixtureMonotoneTrialError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
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
    limits = np.asarray(
        env.getJointGroup("fixture_monotone_nine", list(names)).getLimits().joint_limits
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    labels = [r["stage"] for r in rows]
    report = {
        "tour": tour,
        "scene_setup_s_contended": setup,
        "chains": [],
        "scope": "Native monotone fixture-axis interpolation and generated arm/track spline; no free rail optimization, physical timing or fixture dynamics claim.",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    for factor, label in [
        (factor, label) for factor in args.duration_factor for label in args.chain
    ]:
        result = {
            "first_stage": label,
            "duration_factor": factor,
            "status": "in_progress",
            "stages": [],
        }
        try:
            i = labels.index(label)
            if not label.startswith("fixture:") or (i and labels[i - 1].startswith("fixture:")):
                raise FixtureMonotoneTrialError("selection must name a fixture-chain start")
            end = i + 1
            while end < len(rows) and labels[end].startswith("fixture:"):
                end += 1
            chain = rows[i:end]
            durations = [factor * float(x["duration_s"]) for x in clock["stages"][i:end]]
            q0, q1 = np.asarray(chain[0]["states"]), np.asarray(chain[-1]["states"])
            v0 = (q0[-1] - q0[0]) / durations[0]
            v1 = (q1[-1] - q1[0]) / durations[-1]
            v0[7:] = v1[7:] = 0.0
            a0 = np.zeros(9)
            a1 = np.zeros(9)
            model, nodes, stage_times = build_fixture_multichord(chain, durations, v0, a0, v1, a1)
            _, knots = fixture_knot_times(durations)
            whole_curve = build_jerk_model(knots, v0, a0, v1, a1).curve(nodes)
            target_states = np.asarray(
                [chain[0]["states"][0], *[row["states"][-1] for row in chain]]
            )
            fixture_axes = [
                native.MonotoneFixtureAxis.build(stage_times, target_states[:, k].tolist())
                for k in (7, 8)
            ]
            position_bounds = position_extrema(whole_curve)
            excess = [
                max(
                    limits[k, 0] - position_bounds[k]["minimum"],
                    position_bounds[k]["maximum"] - limits[k, 1],
                    0.0,
                )
                for k in range(7)
            ]
            excess.extend(
                max(
                    limits[k, 0] - float(np.min(target_states[:, k])),
                    float(np.max(target_states[:, k])) - limits[k, 1],
                    0.0,
                )
                for k in (7, 8)
            )
            result.update(
                {
                    "modeled_duration_s": sum(durations),
                    "node_count": len(nodes),
                    "modeled_boundary_velocity_start": v0.tolist(),
                    "modeled_boundary_velocity_end": v1.tolist(),
                    "boundary_acceleration": "zero modeled hypothesis, not observed",
                    "fixture_speed_limits": "not supplied",
                    "raw_position_extrema_excess": excess,
                }
            )
            extrema = [
                velocity_extrema(whole_curve, stage_times[j], stage_times[j + 1])
                for j in range(len(chain))
            ]
            result["stages"] = []
            for j, stage in enumerate(chain):
                ts = np.linspace(stage_times[j], stage_times[j + 1], SAMPLES)
                q, vel, acc, jerk = model.evaluate(ts, nodes, 3)
                for k, axis in zip((7, 8), fixture_axes, strict=True):
                    values = axis.evaluate_many(ts.tolist())
                    q[:, k], vel[:, k], acc[:, k], jerk[:, k] = values.T
                checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
                source = np.asarray(stage["states"])
                baseline = np.column_stack(
                    [
                        np.interp(
                            np.linspace(0, 1, SAMPLES), np.linspace(0, 1, len(source)), source[:, k]
                        )
                        for k in range(9)
                    ]
                )
                peaks = [
                    max(abs(extrema[j][k]["minimum"]), abs(extrema[j][k]["maximum"]))
                    for k in range(7)
                ]
                winding = []
                for k in (7, 8):
                    direction = np.sign(source[-1, k] - source[0, k])
                    winding.append(
                        float(max(0.0, -np.min(direction * vel[:, k])))
                        if direction
                        else float(np.max(np.abs(vel[:, k])))
                    )
                result["stages"].append(
                    {
                        "stage": stage["stage"],
                        "collision_and_position": checked,
                        "actual_robot_speed_ratio_by_axis": (
                            np.asarray(peaks) / np.asarray(speeds)
                        ).tolist(),
                        "max_departure_from_generated_chord_by_axis": np.max(
                            np.abs(q - baseline), axis=0
                        ).tolist(),
                        "target_error_by_axis": np.max(
                            np.abs(q[[0, -1]] - source[[0, -1]]), axis=0
                        ).tolist(),
                        "sampled_peak_jerk_by_axis": np.max(np.abs(jerk), axis=0).tolist(),
                        "fixture_axis_velocity_at_stage_end": vel[-1, 7:].tolist(),
                        "sampled_fixture_winding_violation_native_units_s": winding,
                    }
                )
            result["position_bounds_pass"] = not any(x > SQP_FEASIBILITY_BAND for x in excess)
            result["robot_speed_excess_native_units_s"] = [
                max(
                    max(
                        s["actual_robot_speed_ratio_by_axis"][axis] * speeds[axis] - speeds[axis],
                        0.0,
                    )
                    for s in result["stages"]
                )
                for axis in range(7)
            ]
            result["robot_speed_pass"] = all(
                value <= SQP_FEASIBILITY_BAND
                for value in result["robot_speed_excess_native_units_s"]
            )
            result["robot_speed_check_band_native_units_s"] = SQP_FEASIBILITY_BAND
            parity = PARITY_ULPS * np.finfo(float).eps
            result["target_parity_pass"] = all(
                max(s["target_error_by_axis"]) <= parity for s in result["stages"]
            )
            result["sampled_fixture_winding_pass"] = all(
                max(s["sampled_fixture_winding_violation_native_units_s"]) <= parity
                for s in result["stages"]
            )
            result["status"] = (
                "sampled_candidate_physical_and_process_acceptance_unresolved"
                if result["position_bounds_pass"]
                and result["robot_speed_pass"]
                and result["target_parity_pass"]
                and result["sampled_fixture_winding_pass"]
                else "failed_position_speed_targets_or_winding"
            )
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
        if any(r["status"].startswith("failed") for r in report["chains"])
        else "evaluated_unapproved_chains"
    )
    save()
    if snapshot(scene) != original:
        raise FixtureMonotoneTrialError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "chains": [
                    (r["first_stage"], r["duration_factor"], r["status"], r.get("error"))
                    for r in report["chains"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
