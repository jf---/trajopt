"""Fixed-clock generated probe–transit–probe shared-derivative ablation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from all24_spline_scene import native_log
from duration_policy import scale_transit_clock, validate_fixture_policy
from free_space_transit import _restore_scene
from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene, snapshot
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve
from horizon_search import (
    CandidateInfeasible,
    require_ordered_progress,
    search_shared_rates,
    weighted_jerk,
)
from native_accel_transits import TransitContractError, refine_transits
from process_geometry import assess_process_geometry
from progress_clock import ProgressClockError, compose_process
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Shared grid of normalized endpoint progress rates, not per-tour tuning.
RATE_GRID = (0.5, 1.0, 1.5)
PROCESS_SAMPLES = 257
TRANSIT_SAMPLES = 1025
SQP_FEASIBILITY_BAND = 1e-4  # Existing SQP feasibility input, native joint units.
STATE_ROUNDOFF_ULPS = 128  # Floating-point source-state equality, not a process tolerance.


class HorizonInputError(RuntimeError):
    """The selected generated neighborhood violates the horizon contract."""


def run(scene, segments, clock, speed, artifacts, fixture_policy, output, transit_label):
    """Refine one genuine process–transit–process neighborhood on its fixed clock."""
    before = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    labels = [s["stage"] for s in segments]
    if labels.count(transit_label) != 1:
        raise HorizonInputError("selected transit must exist exactly once")
    index = labels.index(transit_label)
    if index < 1 or index + 1 >= len(segments):
        raise HorizonInputError("a transit requires two adjacent process stages")
    left_stage, transit_stage, right_stage = segments[index - 1 : index + 2]
    if not left_stage["stage"].startswith("withdraw:") or not right_stage["stage"].startswith(
        "approach:"
    ):
        raise HorizonInputError(
            "this ablation requires generated probe withdrawal and approach legs"
        )
    left_probe = int(left_stage["stage"].split(":")[1])
    right_probe = int(right_stage["stage"].split(":")[1])
    fixture = json.loads(fixture_policy.read_text())["fixture"]
    validate_fixture_policy(segments, fixture)
    clock1 = scale_transit_clock(segments, clock, 1.0, fixture["stage_times_s"])
    durations = [float(clock1["stages"][i]["duration_s"]) for i in (index - 1, index, index + 1)]
    baseline = json.loads((artifacts / "tour09-all-weld-blends-assembled.json").read_text())
    if [x["stage"] for x in baseline["stages"]] != labels:
        raise HorizonInputError("baseline stage sequence differs from fresh generated route")
    fixture_values = np.asarray(transit_stage["states"][0], dtype=float)[7:]
    original = []
    curves = []
    for stage, probe, part in ((left_stage, left_probe, 3), (right_stage, right_probe, 0)):
        data = json.loads(
            (
                artifacts / "all-probes-pass1" / f"probe{probe:02d}" / "ik-progress-report.json"
            ).read_text()
        )
        states = np.asarray(data["stage_states"][part], dtype=float).copy()
        stage_q = np.asarray(stage["states"], dtype=float)
        if states.ndim != 2 or states.shape[1] != 7 or not np.isfinite(states).all():
            raise HorizonInputError("generated process geometry needs finite seven-axis states")
        band = STATE_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, float(np.max(np.abs(stage_q))))
        if (
            np.max(np.abs(np.asarray(data["fixture"]) - fixture_values)) > band
            or np.max(np.abs(stage_q[:, 7:] - fixture_values)) > band
        ):
            raise HorizonInputError("fixture changes inside the process horizon")
        if (
            np.max(np.abs(states[0] - stage_q[0, :7])) > band
            or np.max(np.abs(states[-1] - stage_q[-1, :7])) > band
        ):
            raise HorizonInputError("generated process endpoints differ from source stages")
        states[0], states[-1] = stage_q[0, :7], stage_q[-1, :7]
        curves.append(CubicSpline(np.linspace(0.0, 1.0, len(states)), states, axis=0))
        original.append(stage_q)
    if (
        np.max(np.abs(original[0][-1] - np.asarray(transit_stage["states"][0]))) > band
        or np.max(np.abs(original[1][0] - np.asarray(transit_stage["states"][-1]))) > band
    ):
        raise HorizonInputError("nine-axis transit endpoints mismatch neighboring processes")
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
        {"stage": selected[j]["stage"], "duration_s": item["duration_s"]}
        for j, item in enumerate(clock1["stages"])
    ]
    start_state = {str(k): float(v) for k, v in env.getState().joints.items()}
    revision = int(env.getRevision())
    started = perf_counter()
    try:
        env.setState(dict(zip(names[-2:], map(float, fixture_values), strict=True)))
        checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
        limits = np.asarray(
            env.getJointGroup("overlap-horizon-nine", list(names)).getLimits().joint_limits
        )

        def pose(q):
            state = env.getState(
                {
                    **dict(zip(names[7:], map(float, fixture_values), strict=True)),
                    **dict(zip(names[:7], map(float, q), strict=True)),
                }
            )
            return np.linalg.solve(
                np.asarray(state.link_transforms["workpiece_link"].matrix),
                np.asarray(state.link_transforms["nozzle_link"].matrix),
            )

        endpoints = [(pose(x[0, :7]), pose(x[-1, :7])) for x in original]

        def evaluate(left_rate, right_rate):
            candidate_started = perf_counter()
            times = [
                np.linspace(0.0, durations[0], PROCESS_SAMPLES),
                np.linspace(0.0, durations[2], PROCESS_SAMPLES),
            ]
            try:
                process = [
                    compose_process(curves[0], times[0], durations[0], 0.0, left_rate),
                    compose_process(curves[1], times[1], durations[2], right_rate, 0.0),
                ]
            except ProgressClockError as exc:
                raise CandidateInfeasible(str(exc)) from exc
            left, right = process
            left_v, right_v = np.r_[left[1][-1], 0.0, 0.0], np.r_[right[1][0], 0.0, 0.0]
            left_a, right_a = np.r_[left[2][-1], 0.0, 0.0], np.r_[right[2][0], 0.0, 0.0]

            def boundary(*_args, **_kwargs):
                return left_v, right_v, ()

            def acceleration(_index):
                return left_a, right_a

            try:
                with native_log(output / f"candidate-{left_rate:g}-{right_rate:g}.log"):
                    result = refine_transits(
                        env,
                        info,
                        names,
                        selected,
                        {"stages": selected_clock},
                        speed,
                        start_velocity=left_v,
                        end_velocity=right_v,
                        boundary_resolver=boundary,
                        acceleration_resolver=acceleration,
                    )
            except TransitContractError as exc:
                raise CandidateInfeasible(str(exc)) from exc
            if len(result) != 1:
                raise HorizonInputError("expected exactly one optimized transit")
            transit = result[0]
            model = native.QuinticSplineModel.build(
                transit["knots_s"],
                np.asarray(transit["v0"]),
                np.asarray(transit["a0"]),
                np.asarray(transit["v1"]),
                np.asarray(transit["a1"]),
            )
            tt = np.linspace(0.0, durations[1], TRANSIT_SAMPLES)
            nodes = np.asarray(transit["nodes"])
            q, v, a = model.evaluate(tt, nodes, 2)
            mapping = model.map(tt.tolist(), 3)
            j = mapping.basis @ nodes + mapping.offset
            parts = [left, (q, v, a, j), right]
            checks, geometry, scores = [], [], []
            for n, part in enumerate(parts):
                full = np.column_stack((part[0], np.tile(fixture_values, (len(part[0]), 1))))
                try:
                    checks.append(
                        inspect_sampled_curve(checker, full, limits, SQP_FEASIBILITY_BAND)
                    )
                except (CurveCollisionError, CurveLimitError) as exc:
                    raise CandidateInfeasible(
                        f"{[left_stage['stage'], transit_label, right_stage['stage']][n]}: {exc}"
                    ) from exc
                sample_times = [times[0], tt, times[1]][n]
                scores.append(weighted_jerk(sample_times, part[3], speed))
                if n != 1:
                    k = 0 if n == 0 else 1
                    actual = np.asarray([pose(state) for state in part[0]])
                    geometry.append(
                        {
                            "stage": [left_stage["stage"], right_stage["stage"]][k],
                            **assess_process_geometry(*endpoints[k], actual),
                        }
                    )
                    require_ordered_progress(geometry[-1])
            gaps = []
            for first, second in ((left, parts[1]), (parts[1], right)):
                gaps.append(
                    {
                        key: float(np.max(np.abs(first[i][-1] - second[i][0])))
                        for i, key in enumerate(("q", "qd", "qdd"))
                    }
                )
            outer = []
            for z, source, sign in (
                (left, baseline["stages"][index - 1], "first"),
                (right, baseline["stages"][index + 1], "last"),
            ):
                k = 0 if sign == "first" else -1
                outer.append(
                    {
                        key: float(
                            np.max(np.abs(z[i][k] - np.asarray(source[f"{sign}_{key}"])[:7]))
                        )
                        for i, key in enumerate(("q", "qd", "qdd"))
                    }
                )
            return {
                "weighted_jerk": sum(scores),
                "weighted_jerk_by_stage": scores,
                "max_abs_jerk_by_axis": np.max(
                    np.vstack([np.max(np.abs(x[3]), axis=0) for x in parts]), axis=0
                ).tolist(),
                "max_abs_velocity_by_axis": np.max(
                    np.vstack([np.max(np.abs(x[1]), axis=0) for x in parts]), axis=0
                ).tolist(),
                "max_abs_acceleration_by_axis": np.max(
                    np.vstack([np.max(np.abs(x[2]), axis=0) for x in parts]), axis=0
                ).tolist(),
                "process_geometry": geometry,
                "checks": checks,
                "join_gaps": gaps,
                "outer_gaps": outer,
                "transit": transit,
                "elapsed_s": perf_counter() - candidate_started,
            }

        def checkpoint(outcomes):
            (output / "candidates.json").write_text(
                json.dumps(outcomes, indent=2, allow_nan=False) + "\n"
            )

        outcomes, best = search_shared_rates(RATE_GRID, evaluate, checkpoint)
        report = {
            "scope": "One generated fixed-clock probe-transit-probe horizon; bounded discrete shared-rate search, not a continuous optimum or controller timing. Finite checks, hypothetical SearchL hits and unchanged generated process geometry.",
            "stage_sequence": [left_stage["stage"], transit_label, right_stage["stage"]],
            "durations_s": durations,
            "fixture_values": fixture_values.tolist(),
            "rate_grid": RATE_GRID,
            "baseline_rates": [1.0, 1.0],
            "candidates": outcomes,
            "selected": best,
            "elapsed_s": perf_counter() - started,
            "overall_accepted": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        _restore_scene(env, start_state, revision)
        if snapshot(scene) != before:
            raise HorizonInputError("scene changed during horizon ablation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "original-worktree",
        "fresh-route",
        "verified-route",
        "artifacts",
        "fixture-policy",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--transit", required=True)
    parser.add_argument("--timing-note", default="Host contention not characterized")
    args = parser.parse_args()
    libs = require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise HorizonInputError("output path already exists")
    args.output.mkdir(parents=True)
    runtime = {"loaded_libraries": [str(x) for x in libs], "timing_note": args.timing_note}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    started = perf_counter()
    scene = initialize_scene(args.station, args.original_worktree, args.output)
    setup_s = perf_counter() - started
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
    segments, clock, speed = prepare_fresh(
        env,
        program,
        source,
        tours,
        home,
        names,
        raw,
        metadata,
        9,
        json.loads(args.fresh_route.read_text()),
        json.loads(args.verified_route.read_text()),
    )
    result = run(
        scene,
        segments,
        clock,
        speed,
        args.artifacts,
        args.fixture_policy,
        args.output,
        args.transit,
    )
    runtime["scene_setup_s"] = setup_s
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected_rates": result["selected"]["rates"],
                "objective": result["selected"]["weighted_jerk"],
                "elapsed_s": result["elapsed_s"],
                "scene_setup_s": setup_s,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
