"""Evaluate one uniform generated-fixture interpolant over a verified fresh tour."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from continuous_spline_jerk import build_jerk_model
from fixture_multichord import build_fixture_multichord
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing finite actual-curve check per fixture stage.
MACHINE_ULPS = 32768  # Numerical comparison only, not a physical tolerance.


class FixtureTrialError(RuntimeError):
    """A generated fixture chain cannot use this explicit timing hypothesis."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "recovery", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise FixtureTrialError("independently verified generated recovery required")
    if args.output.exists():
        raise FixtureTrialError("output already exists")
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
    limits = np.asarray(
        env.getJointGroup("fixture_multichord_nine", list(names)).getLimits().joint_limits
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    report = {
        "tour": tour,
        "scene_setup_s_contended": setup,
        "chains": [],
        "scope": "Generated target interpolation, modeled clock and boundary hypothesis; finite collision samples; fixture limits and physical boundaries unknown.",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    i = 0
    while i < len(rows):
        if not rows[i]["stage"].startswith("fixture:"):
            i += 1
            continue
        end = i + 1
        while end < len(rows) and rows[end]["stage"].startswith("fixture:"):
            end += 1
        chain = rows[i:end]
        durations = [float(t["duration_s"]) for t in clock["stages"][i:end]]
        result = {"first_stage": chain[0]["stage"], "stages": [], "status": "in_progress"}
        try:
            first = np.asarray(chain[0]["states"], dtype=float)
            last = np.asarray(chain[-1]["states"], dtype=float)
            v0 = (first[-1] - first[0]) / durations[0]
            v1 = (last[-1] - last[0]) / durations[-1]
            # Positioner and tailstock are held on neighboring transit stages.
            v0[7:] = 0.0
            v1[7:] = 0.0
            # Modeled boundary accelerations, never asserted as observed physical data.
            a0 = np.zeros(9)
            a1 = np.zeros(9)
            model, nodes, stage_times = build_fixture_multichord(chain, durations, v0, a0, v1, a1)
            native_times = sorted(
                set(
                    [
                        *stage_times,
                        *[(stage_times[j] + stage_times[j + 1]) / 2 for j in range(len(chain))],
                    ]
                )
            )
            if len(native_times) < 6:
                native_times = sorted(
                    set([*native_times, *np.linspace(0, stage_times[-1], 6).tolist()])
                )
            curve = build_jerk_model(native_times, v0, a0, v1, a1).curve(nodes)
            extrema = position_extrema(curve)
            excess = [
                max(limits[k, 0] - e["minimum"], e["maximum"] - limits[k, 1], 0.0)
                for k, e in enumerate(extrema)
            ]
            result.update(
                {
                    "modeled_v0": v0.tolist(),
                    "modeled_v1": v1.tolist(),
                    "modeled_boundary_acceleration": "zero hypothesis, not observed",
                    "node_count": len(nodes),
                    "raw_position_extrema_excess": excess,
                }
            )
            for j, stage in enumerate(chain):
                ts = np.linspace(stage_times[j], stage_times[j + 1], SAMPLES)
                q, vel, acc = model.evaluate(ts, nodes, 2)
                checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
                generated = np.asarray(stage["states"], dtype=float)
                source = np.column_stack(
                    [
                        np.interp(
                            np.linspace(0, 1, SAMPLES),
                            np.linspace(0, 1, len(generated)),
                            generated[:, k],
                        )
                        for k in range(9)
                    ]
                )
                delta = generated[-1] - generated[0]
                reversal = []
                backtrack = []
                for k in range(9):
                    direction = np.sign(delta[k])
                    reversal.append(
                        float(max(0.0, -np.min(direction * np.diff(q[:, k])))) if direction else 0.0
                    )
                    progress = direction * (q[:, k] - q[0, k])
                    backtrack.append(
                        float(np.max(np.maximum.accumulate(progress) - progress))
                        if direction
                        else 0.0
                    )
                result["stages"].append(
                    {
                        "stage": stage["stage"],
                        "duration_s": durations[j],
                        "collision_and_position": checked,
                        "max_departure_from_generated_chord_by_axis": np.max(
                            np.abs(q - source), axis=0
                        ).tolist(),
                        "max_reverse_sample_step_by_axis": reversal,
                        "max_cumulative_backtrack_by_axis": backtrack,
                        "sampled_robot_speed_ratio": (
                            np.max(np.abs(vel[:, :7]), axis=0) / np.asarray(speeds)
                        ).tolist(),
                        "max_abs_acceleration_by_axis": np.max(np.abs(acc), axis=0).tolist(),
                        "endpoint_target_error_by_axis": np.max(
                            np.abs(q[[0, -1]] - generated[[0, -1]]), axis=0
                        ).tolist(),
                    }
                )
            result["position_bounds_pass"] = not any(e > SQP_FEASIBILITY_BAND for e in excess)
            result["sampled_robot_speed_pass"] = all(
                max(s["sampled_robot_speed_ratio"]) <= 1 + MACHINE_ULPS * np.finfo(float).eps
                for s in result["stages"]
            )
            result["sampled_fixture_winding_pass"] = all(
                max(s["max_cumulative_backtrack_by_axis"][7:]) <= MACHINE_ULPS * np.finfo(float).eps
                for s in result["stages"]
            )
            result["status"] = (
                "sampled_chain_process_and_physical_acceptance_unresolved"
                if result["position_bounds_pass"]
                and result["sampled_robot_speed_pass"]
                and result["sampled_fixture_winding_pass"]
                else "failed_position_speed_or_fixture_winding"
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
        i = end
    report["status"] = (
        "completed_with_failures"
        if any(x["status"].startswith("failed") for x in report["chains"])
        else "evaluated_unapproved_chains"
    )
    save()
    if snapshot(scene) != original:
        raise FixtureTrialError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "chains": len(report["chains"]),
                "failed": sum(x["status"].startswith("failed") for x in report["chains"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
