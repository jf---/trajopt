"""Connect generated fixture chains to adjacent transits in one fresh tour."""

import argparse
import json
import traceback
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage
from actual_stage_contract import Seconds, diagnostic_joins
from actual_tour_assembly import assemble_actual_tour
from continuous_spline_jerk import build_jerk_model
from event_aware_spline_boundaries import require_fixed_fixture_boundary_velocities
from fixture_composite_stage import fixture_composite_stage
from fixture_multichord import build_fixture_multichord
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from native_accel_transits import refine_transits
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries
from spline_velocity_extrema import velocity_extrema
from transit_neighbor_map import transit_neighbor_map
from verified_linear_stages import actual_linear_stage

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent finite collision checks per changed transit.
PARITY_ULPS = 32768  # Binary64 same-curve parity, not physical tolerance.
MAX_POSITION_WITNESS_ROUNDS = 6  # Shared bounded actual-extremum refinement budget.


class FixtureTransitTourError(RuntimeError):
    """The generated tour cannot preserve fixture/transit boundaries."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "recovery", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fixture-duration-factor", type=float, required=True)
    parser.add_argument(
        "--resume-from", type=Path, help="resume previously checked transit results"
    )
    parser.add_argument(
        "--retry-stage",
        action="append",
        default=[],
        help="retry selected failed transits from the checkpoint",
    )
    parser.add_argument(
        "--adaptive-position-witness",
        action="store_true",
        help="refine prior actual position extrema",
    )
    parser.add_argument(
        "--checked-neighborhood",
        type=Path,
        help="include a separately checked generated lead and transit",
    )
    args = parser.parse_args()
    if args.adaptive_position_witness and not args.retry_stage:
        raise FixtureTransitTourError("adaptive position witnesses require a failed-stage retry")
    if args.checked_neighborhood and not args.resume_from:
        raise FixtureTransitTourError("checked neighborhood requires a transit checkpoint")
    if not np.isfinite(args.fixture_duration_factor) or args.fixture_duration_factor <= 0:
        raise FixtureTransitTourError("positive common fixture duration factor required")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise FixtureTransitTourError("verified fresh recovery required")
    if args.output.exists():
        raise FixtureTransitTourError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    original = snapshot(scene)
    env, program, source, tours, names, _, _, _, _, home, info, _ = scene["inputs"]
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    tour = int(recovery["verified"]["tour"])
    rows, original_clock, speeds = prepare_fresh(
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
    clock = deepcopy(original_clock)
    labels = [r["stage"] for r in rows]
    fixture_indices = [i for i, label in enumerate(labels) if label.startswith("fixture:")]
    for i in fixture_indices:
        clock["stages"][i]["duration_s"] *= args.fixture_duration_factor
    report = {
        "tour": tour,
        "scene_setup_s_contended": setup,
        "fixture_duration_factor": args.fixture_duration_factor,
        "fixture_modeled_s_before": sum(
            float(original_clock["stages"][i]["duration_s"]) for i in fixture_indices
        ),
        "fixture_modeled_s_after": sum(
            float(clock["stages"][i]["duration_s"]) for i in fixture_indices
        ),
        "scope": "Fresh generated tour: ordered modeled fixture axes and adjacent optimized transits; unmodified process and remaining transit joint-linear interpolation. Physical boundary derivatives and fixture dynamics unknown.",
        "transits": [],
        "fixture_chains": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    actual = {}
    replacements = {}
    indices_to_refine = set()
    i = 0
    while i < len(rows):
        if not labels[i].startswith("fixture:"):
            if not labels[i].startswith("transit:"):
                actual[i] = actual_linear_stage(
                    labels[i], rows[i]["states"], float(clock["stages"][i]["duration_s"])
                )
            i += 1
            continue
        end = i + 1
        while end < len(rows) and labels[end].startswith("fixture:"):
            end += 1
        if (
            i == 0
            or end == len(rows)
            or not labels[i - 1].startswith("transit:")
            or not labels[end].startswith("transit:")
        ):
            raise FixtureTransitTourError(f"{labels[i]}: missing adjacent transits")
        chain = rows[i:end]
        durations = [float(item["duration_s"]) for item in clock["stages"][i:end]]
        first, last = np.asarray(chain[0]["states"]), np.asarray(chain[-1]["states"])
        v0 = (first[-1] - first[0]) / durations[0]
        v1 = (last[-1] - last[0]) / durations[-1]
        v0[7:] = v1[7:] = 0.0
        a0 = np.zeros(9)
        a1 = np.zeros(9)
        model, nodes, times = build_fixture_multichord(chain, durations, v0, a0, v1, a1)
        targets = np.asarray([chain[0]["states"][0], *[row["states"][-1] for row in chain]])
        axes = tuple(
            native.MonotoneFixtureAxis.build(times, targets[:, k].tolist()) for k in (7, 8)
        )
        for k, j in enumerate(range(i, end)):
            stage = fixture_composite_stage(labels[j], model, nodes, axes, times[k], times[k + 1])
            actual[j] = stage
            replacements[labels[j]] = stage
        indices_to_refine.update((i - 1, end))
        report["fixture_chains"].append(
            {"first_stage": labels[i], "stage_count": end - i, "modeled_duration_s": sum(durations)}
        )
        i = end
    limits = np.asarray(
        env.getJointGroup("fixture_transit_tour_nine", list(names)).getLimits().joint_limits
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    neighborhood_transit = None
    if args.checked_neighborhood:
        evidence = json.loads(args.checked_neighborhood.read_text())
        if (
            evidence["status"] != "sampled_neighborhood_physical_and_process_acceptance_unresolved"
            or not evidence.get("cartesian_track_coordination")
            or evidence["tour"] != tour
            or evidence["fixture_duration_factor"] != args.fixture_duration_factor
        ):
            raise FixtureTransitTourError(
                "checked generated neighborhood or fixture clock mismatch"
            )
        lead_index = labels.index(evidence["lead"])
        transit_index = labels.index(evidence["transit"])
        if transit_index != lead_index + 1 or transit_index not in indices_to_refine:
            raise FixtureTransitTourError("checked lead and selected transit are not adjacent")
        checked_stages = []
        checked_data = []
        for label, data, times_key, index in (
            (evidence["lead"], evidence["lead_model"], "times_s", lead_index),
            (evidence["transit"], evidence["transit_curve"], "knots_s", transit_index),
        ):
            times = data[times_key]
            nodes = np.asarray(data["nodes"], dtype=float)
            source = np.asarray(rows[index]["states"], dtype=float)
            band = (
                PARITY_ULPS * np.finfo(float).eps * max(1.0, float(np.max(np.abs(source[[0, -1]]))))
            )
            if (
                np.max(np.abs(nodes[[0, -1]] - source[[0, -1], :7])) > band
                or not np.allclose(data["fixture"], source[0, 7:], rtol=0, atol=band)
                or abs(float(times[-1]) - float(clock["stages"][index]["duration_s"])) > band
            ):
                raise FixtureTransitTourError(
                    f"{label}: checked source endpoints, fixture or clock differ"
                )
            v0, a0, v1, a1 = (
                np.asarray(data[key], dtype=float) for key in ("v0", "a0", "v1", "a1")
            )
            model = native.QuinticSplineModel.build(times, v0, a0, v1, a1)
            curve = build_jerk_model(times, v0, a0, v1, a1).curve(nodes)
            ts = np.linspace(0.0, float(times[-1]), SAMPLES)
            q, _ = model.evaluate(ts, nodes, 1)
            full = np.column_stack((q, np.tile(data["fixture"], (len(q), 1))))
            collision = inspect_sampled_curve(checker, full, limits, SQP_FEASIBILITY_BAND)
            position = position_extrema(curve)
            velocity = velocity_extrema(curve, 0.0, float(times[-1]))
            position_excess = [
                max(limits[k, 0] - item["minimum"], item["maximum"] - limits[k, 1], 0.0)
                for k, item in enumerate(position)
            ]
            speed_excess = [
                max(abs(item["minimum"]) - speeds[k], abs(item["maximum"]) - speeds[k], 0.0)
                for k, item in enumerate(velocity)
            ]
            if any(value > SQP_FEASIBILITY_BAND for value in [*position_excess, *speed_excess]):
                raise FixtureTransitTourError(
                    f"{label}: checked curve fails actual position or speed extrema"
                )
            checked_stages.append(
                transit_stage(label, float(times[-1]), model, nodes, data["fixture"])
            )
            checked_data.append(
                {
                    "stage": label,
                    "collision_and_position": collision,
                    "raw_position_extrema_excess": position_excess,
                    "raw_velocity_extrema_excess": speed_excess,
                }
            )
        actual[lead_index] = checked_stages[0]
        replacements[evidence["lead"]] = checked_stages[0]
        neighborhood_transit = {
            "stage": evidence["transit"],
            "status": "sampled_transit_fixture_join_checked",
            "optimized_curve": evidence["transit_curve"],
            "checked_neighborhood_recheck": checked_data,
            "process_boundary": "checked generated lead; physical derivatives unknown",
            "lead_geometry_evidence": evidence["evaluations"][0]["generated_lead_geometry"],
            "modeled_lead_feed_range_m_s": evidence["evaluations"][0][
                "modeled_tool_feed_range_m_s"
            ],
        }
        report["checked_neighborhood"] = {
            "lead": evidence["lead"],
            "transit": evidence["transit"],
            "rechecked_curves": checked_data,
            "process_acceptance": "unresolved",
        }
    neighbors = transit_neighbor_map(rows, clock["stages"], actual)
    fixture_join_diagnostics = []
    prior = {}
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        if (
            saved["tour"] != tour
            or saved["fixture_duration_factor"] != args.fixture_duration_factor
        ):
            raise FixtureTransitTourError("resumed tour or modeled fixture clock differs")
        prior = {item["stage"]: item for item in saved["transits"]}
        if len(prior) != len(saved["transits"]) or set(prior) != {
            labels[index] for index in indices_to_refine
        }:
            raise FixtureTransitTourError("resume requires every selected transit exactly once")
        if any(
            stage not in prior or prior[stage]["status"] != "failed" for stage in args.retry_stage
        ):
            raise FixtureTransitTourError("only previously failed transits may be retried")
        if neighborhood_transit is not None:
            old = prior.get(neighborhood_transit["stage"])
            if old is None or old["status"] != "failed":
                raise FixtureTransitTourError(
                    "checked neighborhood must replace an explicitly failed transit"
                )
            neighborhood_transit["previous_failure"] = {
                "error": old.get("error"),
                "status": old["status"],
            }
            prior[neighborhood_transit["stage"]] = neighborhood_transit
    elif args.retry_stage:
        raise FixtureTransitTourError("retry requires a checkpoint")
    for index in sorted(indices_to_refine):
        label = labels[index]
        row = rows[index]
        timing = clock["stages"][index]
        result = {"stage": label, "status": "in_progress"}
        try:
            left, right, events = neighbors[index]
            if label in prior and label not in args.retry_stage:
                result = prior[label]
                if result["status"] == "sampled_transit_fixture_join_checked":
                    transit = result["optimized_curve"]
                    nodes = np.asarray(transit["nodes"])
                    expected = np.asarray(row["states"])[[0, -1], :7]
                    band = (
                        PARITY_ULPS
                        * np.finfo(float).eps
                        * max(1.0, float(np.max(np.abs(expected))))
                    )
                    if np.max(np.abs(nodes[[0, -1]] - expected)) > band or not np.allclose(
                        transit["fixture"], np.asarray(row["states"])[0, 7:], rtol=0, atol=band
                    ):
                        raise FixtureTransitTourError("resumed transit endpoints or fixture differ")
                    model = native.QuinticSplineModel.build(
                        transit["knots_s"],
                        np.asarray(transit["v0"]),
                        np.asarray(transit["a0"]),
                        np.asarray(transit["v1"]),
                        np.asarray(transit["a1"]),
                    )
                    stage = transit_stage(
                        label, transit["duration_s"], model, nodes, transit["fixture"]
                    )
                    resume_checks = []
                    if left is not None and left.label.startswith("fixture:"):
                        resume_checks.extend(j._asdict() for j in diagnostic_joins([left, stage]))
                    if right is not None and right.label.startswith("fixture:"):
                        resume_checks.extend(j._asdict() for j in diagnostic_joins([stage, right]))
                    if neighborhood_transit is not None and label == neighborhood_transit["stage"]:
                        if left is None:
                            raise FixtureTransitTourError("checked lead boundary missing")
                        resume_checks.extend(j._asdict() for j in diagnostic_joins([left, stage]))
                    if any(
                        j[k] is None or max(map(abs, j[k])) > band
                        for j in resume_checks
                        for k in ("position_delta", "velocity_delta", "acceleration_delta")
                    ):
                        raise FixtureTransitTourError(
                            "resumed actual fixture or checked lead derivatives differ"
                        )
                    fixture_join_diagnostics.extend(resume_checks)
                    replacements[label] = stage
                report["transits"].append(result)
                continue
            if left is None or right is None:
                raise FixtureTransitTourError("unknown outer physical boundary velocity")
            begin = left.evaluate(left.duration)
            finish = right.evaluate(Seconds(0.0))
            if begin.velocity is None or finish.velocity is None:
                raise FixtureTransitTourError("neighboring modeled velocity unavailable")
            v0, v1 = np.asarray(begin.velocity), np.asarray(finish.velocity)
            require_fixed_fixture_boundary_velocities(v0, v1)
            left_fixture = left.label.startswith("fixture:")
            right_fixture = right.label.startswith("fixture:")
            a0 = (
                np.asarray(begin.acceleration)
                if left_fixture and begin.acceleration is not None
                else None
            )
            a1 = (
                np.asarray(finish.acceleration)
                if right_fixture and finish.acceleration is not None
                else None
            )
            if (left_fixture and a0 is None) or (right_fixture and a1 is None):
                raise FixtureTransitTourError("actual fixture boundary acceleration unavailable")

            def boundaries(_rows, _timing, _index, *, start_velocity, end_velocity):
                return np.asarray(start_velocity), np.asarray(end_velocity), events

            def accelerations(_index):
                return a0, a1

            witnesses = []
            seed = None
            seed_accelerations = None
            if args.adaptive_position_witness:
                previous = prior[label]
                if "actual_position_extrema" not in previous or "optimized_curve" not in previous:
                    raise FixtureTransitTourError(
                        "prior actual extrema and rejected curve required"
                    )
                for k, extreme in enumerate(previous["actual_position_extrema"]):
                    if limits[k, 0] - extreme["minimum"] > SQP_FEASIBILITY_BAND:
                        witnesses.append(extreme["minimum_time_s"])
                    if extreme["maximum"] - limits[k, 1] > SQP_FEASIBILITY_BAND:
                        witnesses.append(extreme["maximum_time_s"])
                if not witnesses:
                    raise FixtureTransitTourError(
                        "prior candidate has no configured position violation"
                    )
                seed = previous["optimized_curve"]["nodes"]
                seed_accelerations = [
                    previous["optimized_curve"]["a0"],
                    previous["optimized_curve"]["a1"],
                ]
            result["position_witness_attempts"] = []
            for attempt in range(
                MAX_POSITION_WITNESS_ROUNDS if args.adaptive_position_witness else 1
            ):
                solved = refine_transits(
                    env,
                    info,
                    names,
                    [row],
                    {"stages": [timing]},
                    speeds,
                    start_velocity=v0,
                    end_velocity=v1,
                    boundary_resolver=boundaries,
                    acceleration_resolver=accelerations,
                    position_witness_times_s=witnesses,
                    seed_positions=seed,
                    seed_accelerations=seed_accelerations,
                )
                if len(solved) != 1 or solved[0]["status"] != "SQPStatus.NLP_CONVERGED":
                    raise FixtureTransitTourError("native transit solver did not converge")
                transit = solved[0]
                attempted_curve = build_jerk_model(
                    transit["knots_s"],
                    np.asarray(transit["v0"]),
                    np.asarray(transit["a0"]),
                    np.asarray(transit["v1"]),
                    np.asarray(transit["a1"]),
                ).curve(np.asarray(transit["nodes"]))
                actual_extrema = position_extrema(attempted_curve)
                new_times = []
                for k, extreme in enumerate(actual_extrema):
                    if limits[k, 0] - extreme["minimum"] > SQP_FEASIBILITY_BAND:
                        new_times.append(extreme["minimum_time_s"])
                    if extreme["maximum"] - limits[k, 1] > SQP_FEASIBILITY_BAND:
                        new_times.append(extreme["maximum_time_s"])
                result["position_witness_attempts"].append(
                    {
                        "round": attempt,
                        "witnesses_s": list(witnesses),
                        "actual_extrema": actual_extrema,
                        "solver_status": transit["status"],
                        "solver_s_contended": transit["solver_s"],
                    }
                )
                if not args.adaptive_position_witness or not new_times:
                    break
                enlarged = sorted(set(witnesses + new_times))
                if len(enlarged) == len(witnesses):
                    raise FixtureTransitTourError("actual position witness refinement stagnated")
                witnesses = enlarged
                seed = transit["nodes"]
                seed_accelerations = [transit["a0"], transit["a1"]]
            else:
                raise FixtureTransitTourError("actual position witness budget exhausted")
            result["optimized_curve"] = transit
            model = native.QuinticSplineModel.build(
                transit["knots_s"],
                np.asarray(transit["v0"]),
                np.asarray(transit["a0"]),
                np.asarray(transit["v1"]),
                np.asarray(transit["a1"]),
            )
            curve = build_jerk_model(
                transit["knots_s"],
                np.asarray(transit["v0"]),
                np.asarray(transit["a0"]),
                np.asarray(transit["v1"]),
                np.asarray(transit["a1"]),
            ).curve(np.asarray(transit["nodes"]))
            ts = np.linspace(0, transit["duration_s"], SAMPLES)
            q, velocity = model.evaluate(ts, np.asarray(transit["nodes"]), 1)
            full = np.column_stack((q, np.tile(transit["fixture"], (len(q), 1))))
            extrema = position_extrema(curve)
            excess = [
                max(limits[k, 0] - x["minimum"], x["maximum"] - limits[k, 1], 0.0)
                for k, x in enumerate(extrema)
            ]
            result["actual_position_extrema"] = extrema
            result["raw_position_extrema_excess"] = excess
            collision = inspect_sampled_curve(checker, full, limits, SQP_FEASIBILITY_BAND)
            speed_excess = np.maximum(np.max(np.abs(velocity), axis=0) - np.asarray(speeds), 0.0)
            if any(x > SQP_FEASIBILITY_BAND for x in excess) or np.any(
                speed_excess > SQP_FEASIBILITY_BAND
            ):
                raise FixtureTransitTourError(
                    "actual position extrema or sampled sourced speed exceeds configured band"
                )
            stage = transit_stage(
                label, transit["duration_s"], model, transit["nodes"], transit["fixture"]
            )
            checks = []
            if left_fixture:
                checks += [j._asdict() for j in diagnostic_joins([left, stage])]
            if right_fixture:
                checks += [j._asdict() for j in diagnostic_joins([stage, right])]
            band = PARITY_ULPS * np.finfo(float).eps
            if any(
                j[k] is None or max(map(abs, j[k])) > band
                for j in checks
                for k in ("position_delta", "velocity_delta", "acceleration_delta")
            ):
                raise FixtureTransitTourError("actual fixture/transit derivatives do not match")
            fixture_join_diagnostics.extend(checks)
            replacements[label] = stage
            result.update(
                {
                    "status": "sampled_transit_fixture_join_checked",
                    "solver_status": transit["status"],
                    "solver_s_contended": transit["solver_s"],
                    "collision_and_position": collision,
                    "raw_position_extrema_excess": excess,
                    "sampled_speed_excess_native_units_s": speed_excess.tolist(),
                    "process_boundary": "modeled joint-linear velocity; physical velocity and acceleration unknown",
                    "optimized_curve": transit,
                }
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
        report["transits"].append(result)
        save()
    path, corners = assemble_actual_tour(rows, clock, replacements)
    joins = [item._asdict() for item in diagnostic_joins(path)]
    band = PARITY_ULPS * np.finfo(float).eps
    report["assembled_stage_count"] = len(path)
    report["joint_linear_corners"] = len(corners)
    report["modeled_join_count"] = len(joins)
    report["modeled_join_velocity_unresolved"] = int(
        sum(
            bool(j["velocity_delta"] is None or max(map(abs, j["velocity_delta"])) > band)
            for j in joins
        )
    )
    report["modeled_join_acceleration_unresolved"] = int(
        sum(
            bool(j["acceleration_delta"] is None or max(map(abs, j["acceleration_delta"])) > band)
            for j in joins
        )
    )
    report["fixture_join_diagnostics"] = fixture_join_diagnostics
    report["unchanged_generated_stages"] = (
        "original independently sampled path; not rechecked at 1025 samples in this experiment"
    )
    report["status"] = (
        "partial_tour_with_failed_transits_and_unresolved_process"
        if any(t["status"] == "failed" for t in report["transits"])
        else "fixture_transits_checked_process_continuity_unresolved"
    )
    save()
    if snapshot(scene) != original:
        raise FixtureTransitTourError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "fixtures": len(report["fixture_chains"]),
                "transits": len(report["transits"]),
                "failed": sum(t["status"] == "failed" for t in report["transits"]),
                "corners": len(corners),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
