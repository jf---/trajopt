"""Connect one generated weld cluster to its adjacent transits and verify actual curves."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage, weld_stage
from actual_stage_contract import Seconds, diagnostic_joins
from cubic_position_extrema import cubic_position_extrema
from fixture_easing_policy import ease_fixture_chains
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from native_accel_transits import refine_transits
from probe_process_inputs import extract_process_tour
from probe_source_join_trial import pose_matrix
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import rotation_errors
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from weld_only_inputs import WeldBead, extract_tour

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent collision sampling policy.
PARITY_ULPS = 32768  # Same-curve numerical boundary comparison, not process tolerance.


class WeldNeighborhoodError(RuntimeError):
    """The selected generated neighborhood cannot be connected or checked."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "weld-evidence",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    welds = json.loads(args.weld_evidence.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour" or len(welds["beads"]) != 1:
        raise WeldNeighborhoodError(
            "verified fresh recovery and one explicit weld cluster required"
        )
    bead = welds["beads"][0]
    if bead["status"] != "sampled_exploratory_weld_only":
        raise WeldNeighborhoodError("weld candidate did not pass its initial samples")
    if args.output.exists():
        raise WeldNeighborhoodError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    original = snapshot(scene)
    env, program, source, tours, names, _, _, _, _, home, info, _ = scene["inputs"]
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
    labels = [r["stage"] for r in rows]
    weld_labels = [s["stage"] for s in bead["stages"]]
    first = labels.index(weld_labels[0])
    last = first + len(weld_labels) - 1
    if labels[first : last + 1] != weld_labels or first < 2 or last + 2 >= len(rows):
        raise WeldNeighborhoodError("weld cluster is not contiguous with both neighbors")
    if not labels[first - 1].startswith("transit:") or not labels[last + 1].startswith("transit:"):
        raise WeldNeighborhoodError("two adjacent generated transits are required")
    fixture_start = last + 2
    fixture_end = fixture_start
    while fixture_end < len(rows) and labels[fixture_end].startswith("fixture:"):
        fixture_end += 1
    if fixture_end == fixture_start:
        raise WeldNeighborhoodError("following fixture curve is unavailable")
    _, fixture_stages = ease_fixture_chains(
        rows[fixture_start:fixture_end],
        {"stages": clock["stages"][fixture_start:fixture_end]},
    )
    times = np.asarray(bead["times_s"], dtype=float)
    nodes = np.asarray(bead["joint_states_7"], dtype=float)
    fixture = tuple(bead["fixture"])
    expected = np.asarray([rows[first - 1]["states"][-1][:7], rows[last + 1]["states"][0][:7]])
    mismatch = np.abs(nodes[[0, -1]] - expected)
    parity = PARITY_ULPS * np.finfo(float).eps * max(1.0, float(np.max(np.abs(expected))))
    if np.max(mismatch) > parity:
        raise WeldNeighborhoodError("weld and generated transit endpoint states differ")
    nodes[[0, -1]] = expected
    curve = CubicSpline(times, nodes, axis=0)
    durations = [float(clock["stages"][i]["duration_s"]) for i in range(first, last + 1)]
    edges = np.r_[0.0, np.cumsum(durations)]
    weld_stages = [
        weld_stage(labels[i], durations[i - first], curve, float(edges[i - first]), fixture)
        for i in range(first, last + 1)
    ]
    # The left process still uses its explicit generated joint-linear clock.
    preceding = np.asarray(rows[first - 2]["states"], dtype=float)
    preceding_duration = float(clock["stages"][first - 2]["duration_s"])
    if preceding_duration <= 0:
        raise WeldNeighborhoodError("preceding modeled process duration must be positive")
    preceding_v = (preceding[-1] - preceding[-2]) * (len(preceding) - 1) / preceding_duration
    next_fixture = fixture_stages[labels[last + 2]].evaluate(Seconds(0.0))
    left_end = weld_stages[0].evaluate(Seconds(0.0))
    right_start = weld_stages[-1].evaluate(weld_stages[-1].duration)
    if any(
        x is None
        for x in (
            left_end.velocity,
            left_end.acceleration,
            right_start.velocity,
            right_start.acceleration,
            next_fixture.velocity,
            next_fixture.acceleration,
        )
    ):
        raise WeldNeighborhoodError("explicit weld and fixture derivatives required")
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "tour": tour,
        "source_segment": bead["source_segment"],
        "stages": [],
        "outer_boundary_provenance": {
            "left_velocity": "generated joint-linear withdrawal secant, not observed",
            "left_acceleration": "free variable; physical value unknown",
            "right": "modeled eased fixture start; fixture dynamics unsourced",
        },
        "weld_geometry": "exploratory source-orientation targets and Cartesian corner blends, manufacturing acceptance unknown",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        transits = []
        for index, v0, v1, a0, a1 in (
            (
                first - 1,
                preceding_v,
                np.asarray(left_end.velocity),
                None,
                np.asarray(left_end.acceleration),
            ),
            (
                last + 1,
                np.asarray(right_start.velocity),
                np.asarray(next_fixture.velocity),
                np.asarray(right_start.acceleration),
                np.asarray(next_fixture.acceleration),
            ),
        ):
            row, timing = rows[index], clock["stages"][index]

            def boundary(_rows, _timing, _index, *, start_velocity, end_velocity):
                return np.asarray(start_velocity), np.asarray(end_velocity), []

            def accelerations(_index):
                return a0, a1

            result = refine_transits(
                env,
                info,
                names,
                [row],
                {"stages": [timing]},
                speeds,
                start_velocity=v0,
                end_velocity=v1,
                boundary_resolver=boundary,
                acceleration_resolver=accelerations,
            )
            if len(result) != 1 or result[0]["status"] != "SQPStatus.NLP_CONVERGED":
                raise WeldNeighborhoodError(
                    f"{row['stage']}: native solver did not converge: {result}"
                )
            transits.append(result[0])
        report["transits"] = transits
        stages = []
        for item in (transits[0], transits[1]):
            model = native.QuinticSplineModel.build(
                item["knots_s"],
                np.asarray(item["v0"]),
                np.asarray(item["a0"]),
                np.asarray(item["v1"]),
                np.asarray(item["a1"]),
            )
            stages.append(
                transit_stage(
                    item["stage"], item["duration_s"], model, item["nodes"], item["fixture"]
                )
            )
        path = [
            stages[0],
            *weld_stages,
            stages[1],
            *[fixture_stages[labels[i]] for i in range(fixture_start, fixture_end)],
        ]
        report["modeled_joins"] = [x._asdict() for x in diagnostic_joins(path)]
        limits = np.asarray(
            env.getJointGroup("weld_neighborhood_nine", list(names)).getLimits().joint_limits
        )
        checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
        task = extract_tour(source, tours, tour, "workpiece_link", home=home)
        process = extract_process_tour(program, task, probe_metadata_frame="workpiece_link")
        source_bead = next(
            item
            for item in process.operations
            if isinstance(item, WeldBead) and item.segment == bead["source_segment"]
        )
        source_poses = np.asarray([pose_matrix(item) for item in source_bead.poses])

        def pose(q):
            state = env.getState(dict(zip(names, map(float, q), strict=True)))
            return np.linalg.solve(
                np.asarray(state.link_transforms["workpiece_link"].matrix),
                np.asarray(state.link_transforms["nozzle_link"].matrix),
            )

        expected_poses = (
            [pose(np.asarray(rows[first]["states"][0]))]
            + list(source_poses)
            + [pose(np.asarray(rows[last]["states"][-1]))]
        )
        for position, stage in enumerate(path):
            ts = np.linspace(0, float(stage.duration), SAMPLES)
            states = [stage.evaluate(Seconds(float(t))) for t in ts]
            q = np.asarray([s.position for s in states])
            vel = np.asarray([s.velocity[:7] for s in states])
            checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            if stage.label.startswith("transit:"):
                item = next(t for t in transits if t["stage"] == stage.label)
                from continuous_spline_jerk import build_jerk_model

                actual_curve = build_jerk_model(
                    item["knots_s"],
                    np.asarray(item["v0"]),
                    np.asarray(item["a0"]),
                    np.asarray(item["v1"]),
                    np.asarray(item["a1"]),
                ).curve(np.asarray(item["nodes"]))
                extrema = position_extrema(actual_curve)
            elif stage.label.startswith("weld:"):
                extrema = cubic_position_extrema(curve)
            else:
                source_states = np.asarray(rows[labels.index(stage.label)]["states"], dtype=float)
                extrema = [
                    {
                        "minimum": float(np.min(source_states[:, i])),
                        "maximum": float(np.max(source_states[:, i])),
                    }
                    for i in range(9)
                ]
            excess = [
                max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                for i, e in enumerate(extrema)
            ]
            ratios = np.max(np.abs(vel), axis=0) / np.asarray(speeds)
            if any(e > SQP_FEASIBILITY_BAND for e in excess) or np.any(
                ratios > 1 + PARITY_ULPS * np.finfo(float).eps
            ):
                raise WeldNeighborhoodError(f"{stage.label}: actual limits or sampled speed failed")
            entry = {
                "stage": stage.label,
                "duration_s": float(stage.duration),
                "collision_and_position": checked,
                "raw_position_extrema_excess": excess,
                "sampled_speed_ratio": ratios.tolist(),
            }
            if stage.label.startswith("weld:"):
                j = position - 1
                actual = np.asarray([pose(point) for point in q])
                entry["kind"] = (
                    "generated_lead" if j in (0, len(weld_stages) - 1) else "source_bead"
                )
                entry["geometry"] = assess_process_geometry(
                    expected_poses[j], expected_poses[j + 1], actual
                )
                feed = np.linalg.norm(np.gradient(actual[:, :3, 3], ts, axis=0), axis=1)
                entry["modeled_feed_range_m_s"] = [float(np.min(feed)), float(np.max(feed))]
                entry["source_endpoint_position_error_m"] = [
                    float(np.linalg.norm(actual[k, :3, 3] - expected_poses[j + e][:3, 3]))
                    for e, k in enumerate((0, -1))
                ]
                entry["source_endpoint_orientation_error_rad"] = [
                    float(
                        rotation_errors(
                            expected_poses[j + e][None, :3, :3], actual[k][None, :3, :3]
                        )[0]
                    )
                    for e, k in enumerate((0, -1))
                ]
            report["stages"].append(entry)
            save()
        band = PARITY_ULPS * np.finfo(float).eps
        report["unresolved_modeled_joins"] = [
            j
            for j in report["modeled_joins"]
            if any(
                j[k] is None or max(map(abs, j[k])) > band
                for k in ("position_delta", "velocity_delta", "acceleration_delta")
            )
        ]
        report["unresolved_source_bead_order"] = [
            e["stage"]
            for e in report["stages"]
            if e.get("kind") == "source_bead" and e["geometry"]["max_backward_distance_m"] > band
        ]
        report["status"] = (
            "failed_unresolved_modeled_joins_or_bead_order"
            if report["unresolved_modeled_joins"] or report["unresolved_source_bead_order"]
            else "sampled_neighborhood_physical_and_process_acceptance_unresolved"
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    save()
    if snapshot(scene) != original:
        raise WeldNeighborhoodError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "stages_checked": len(report["stages"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
