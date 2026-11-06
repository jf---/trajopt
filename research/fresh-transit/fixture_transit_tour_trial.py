"""Connect generated fixture chains to adjacent transits in one fresh tour."""

import argparse
import json
import resource
import sys
import traceback
from copy import deepcopy
from os import getpid
from pathlib import Path
from time import perf_counter, process_time

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage, weld_stage
from actual_stage_contract import Seconds, StationaryEvent, diagnostic_joins
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
from native_free_boundary_transits import refine_transits as refine_free_boundary_transits
from probe_search_group_parallel import split_ranges
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries
from spline_velocity_extrema import velocity_extrema
from transit_neighbor_map import transit_neighbor_map
from verified_linear_stages import actual_linear_stage
from weld_batch_adapter import (
    adapt_weld_batch,
    canonicalize_weld_transit_endpoints,
    weld_neighbor_indices,
)
from weld_tangent_leads import fit_lead

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
        "--collision-witness-file",
        type=Path,
        help="explicit actual-curve collision witness times for failed transit retries",
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
    parser.add_argument(
        "--probe-batch", type=Path, help="include independently checked generated probe groups"
    )
    parser.add_argument("--weld-batch", type=Path, help="include checked generated weld groups")
    parser.add_argument(
        "--weld-transit-results", type=Path, help="reuse checked weld-adjacent transits"
    )
    parser.add_argument("--weld-partitions", type=int, help="number of weld transit workers")
    parser.add_argument("--weld-partition", type=int, help="zero-based weld worker partition")
    parser.add_argument(
        "--reuse-modeled-outer", type=Path, help="reuse checked modeled outer transits"
    )
    parser.add_argument(
        "--verify-partitions", type=int, help="number of independent tour-check workers"
    )
    parser.add_argument(
        "--verify-partition", type=int, help="zero-based tour-check worker partition"
    )
    parser.add_argument(
        "--probe-partitions", type=int, help="number of independent transit workers"
    )
    parser.add_argument("--probe-partition", type=int, help="zero-based worker partition")
    parser.add_argument(
        "--probe-transit-results", type=Path, help="reuse disjoint checked probe transit results"
    )
    parser.add_argument(
        "--free-outer-boundary",
        action="store_true",
        help="optimize unknown modeled outer transit derivatives",
    )
    parser.add_argument(
        "--outer-trust-box", type=float, help="initial modeled outer solver trust box"
    )
    parser.add_argument(
        "--outer-speed-preference-per-s3",
        type=float,
        default=0.0,
        help="soft normalized modeled outer speed weight in inverse seconds cubed",
    )
    args = parser.parse_args()
    if args.adaptive_position_witness and not args.retry_stage:
        raise FixtureTransitTourError("adaptive position witnesses require a failed-stage retry")
    collision_witnesses = (
        {}
        if args.collision_witness_file is None
        else json.loads(args.collision_witness_file.read_text())
    )
    if collision_witnesses and set(collision_witnesses) != set(args.retry_stage):
        raise FixtureTransitTourError(
            "collision witnesses must cover the explicit failed-stage retries"
        )
    if args.checked_neighborhood and not args.resume_from:
        raise FixtureTransitTourError("checked neighborhood requires a transit checkpoint")
    if args.probe_batch and not args.resume_from:
        raise FixtureTransitTourError("checked probe groups require a transit checkpoint")
    if (args.probe_partitions is None) != (args.probe_partition is None):
        raise FixtureTransitTourError("both worker partition arguments are required")
    if (args.weld_partitions is None) != (args.weld_partition is None):
        raise FixtureTransitTourError("both weld worker partition arguments are required")
    if (args.verify_partitions is None) != (args.verify_partition is None):
        raise FixtureTransitTourError(
            "both full-tour verification partition arguments are required"
        )
    if args.reuse_modeled_outer and not args.free_outer_boundary:
        raise FixtureTransitTourError("reusing modeled outer transits requires explicit opt-in")
    if args.verify_partitions is not None and (
        args.probe_partitions is not None or args.weld_partitions is not None
    ):
        raise FixtureTransitTourError(
            "verification and optimization worker partitions cannot be combined"
        )
    if args.probe_partitions is not None and args.weld_partitions is not None:
        raise FixtureTransitTourError("one worker partition policy at a time required")
    if (args.weld_partitions is not None or args.weld_transit_results) and not args.weld_batch:
        raise FixtureTransitTourError("weld workers require checked weld batch")
    if (args.probe_partitions is not None or args.probe_transit_results) and not args.probe_batch:
        raise FixtureTransitTourError("probe transit partitions require a checked probe batch")
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
        "scope": "Fresh generated tour: ordered modeled fixture axes, supplied checked probe and weld groups, and adjacent optimized transits; remaining process and transit stages use generated joint-linear interpolation. Physical boundary derivatives and fixture dynamics unknown.",
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
    fixture_indices_to_refine = set(indices_to_refine)
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
    probe_indices_to_refine = set()
    checked_probe_labels = set()
    if args.probe_batch:
        batch = json.loads(args.probe_batch.read_text())
        if batch["tour"] != tour or any(
            g["status"] != "sampled_group_physical_and_process_acceptance_unresolved"
            for g in batch["groups"]
        ):
            raise FixtureTransitTourError("probe batch contains a failed or different-tour group")
        if not batch["groups"]:
            raise FixtureTransitTourError("probe batch is empty")
        for group in batch["groups"]:
            phases = group["process"]
            if len(phases) != 4:
                raise FixtureTransitTourError("each probe requires four ordered phases")
            start = labels.index(phases[0]["stage"])
            if [x["stage"] for x in phases] != labels[start : start + 4]:
                raise FixtureTransitTourError("probe phase ordering differs")
            if (
                start == 0
                or start + 4 >= len(rows)
                or not labels[start - 1].startswith("transit:")
                or not labels[start + 4].startswith("transit:")
            ):
                raise FixtureTransitTourError("probe group lacks both adjacent generated transits")
            for offset, phase in enumerate(phases):
                index = start + offset
                source = np.asarray(rows[index]["states"], dtype=float)
                nodes = np.asarray(phase["ik_states"], dtype=float)
                band = (
                    PARITY_ULPS
                    * np.finfo(float).eps
                    * max(1.0, float(np.max(np.abs(source[[0, -1]]))))
                )
                if (
                    nodes.ndim != 2
                    or nodes.shape[1] != 7
                    or np.max(np.abs(nodes[[0, -1]] - source[[0, -1], :7])) > band
                ):
                    raise FixtureTransitTourError(
                        f"{phase['stage']}: generated probe endpoints differ"
                    )
                fixture = tuple(source[0, 7:])
                duration = float(phase["duration_s"])
                times = np.linspace(0.0, duration, len(nodes))
                v0, a0, v1, a1 = (
                    np.asarray(phase[key], dtype=float) for key in ("v0", "a0", "v1", "a1")
                )
                if offset == 1:
                    model = native.QuinticSplineModel.build(times.tolist(), v0, a0, v1, a1)
                    stage = transit_stage(phase["stage"], duration, model, nodes, fixture)
                else:
                    curve = fit_lead(times, nodes, (v0, a0), (v1, a1))
                    stage = weld_stage(phase["stage"], duration, curve, 0.0, fixture)
                actual[index] = stage
                replacements[phase["stage"]] = stage
                checked_probe_labels.add(phase["stage"])
                rows[index] = {
                    "stage": phase["stage"],
                    "states": np.column_stack((nodes, np.tile(fixture, (len(nodes), 1)))).tolist(),
                }
                clock["stages"][index]["duration_s"] = duration
            probe_indices_to_refine.update((start - 1, start + 4))
        indices_to_refine.update(probe_indices_to_refine)
        report["checked_probe_batch"] = {
            "probe_start": batch["probe_start"],
            "probe_count": batch["probe_count"],
            "group_count": len(batch["groups"]),
            "scope": "actual group curves were independently sampled in the explicit input batch; process acceptance unresolved",
        }
    checked_weld_labels = set()
    weld_indices_to_refine = set()
    if args.weld_batch:
        checked_weld, unresolved = adapt_weld_batch(
            rows, clock, json.loads(args.weld_batch.read_text()), tour
        )
        weld_indices_to_refine = weld_neighbor_indices(rows, checked_weld)
        canonical_weld_boundaries = canonicalize_weld_transit_endpoints(rows, checked_weld)
        for index, (stage, states) in checked_weld.items():
            actual[index] = stage
            replacements[stage.label] = stage
            checked_weld_labels.add(stage.label)
            rows[index] = {"stage": stage.label, "states": states.tolist()}
        indices_to_refine.update(weld_indices_to_refine)
        report["checked_weld_batch"] = {
            "stages": len(checked_weld),
            "unresolved_groups": unresolved,
            "canonical_transit_boundaries": canonical_weld_boundaries,
            "scope": "Actual input curves independently sampled; process acceptance unresolved.",
        }
    neighbors = transit_neighbor_map(rows, clock["stages"], actual)
    if args.free_outer_boundary:
        outer = [i for i, (left, right, _) in neighbors.items() if left is None or right is None]
        indices_to_refine.update(outer)
        report["free_outer_boundary_stages"] = [labels[i] for i in outer]
    worker_indices = None
    if args.probe_partitions is not None:
        ranges = split_ranges(0, len(probe_indices_to_refine), args.probe_partitions)
        if args.probe_partition < 0 or args.probe_partition >= len(ranges):
            raise FixtureTransitTourError("worker partition is outside the configured range")
        start, count = ranges[args.probe_partition]
        worker_indices = sorted(probe_indices_to_refine)[start : start + count]
        report["worker_partition"] = args.probe_partition
        report["worker_partitions"] = args.probe_partitions
    if args.weld_partitions is not None:
        ranges = split_ranges(0, len(weld_indices_to_refine), args.weld_partitions)
        if args.weld_partition < 0 or args.weld_partition >= len(ranges):
            raise FixtureTransitTourError("weld worker partition is outside the configured range")
        start, count = ranges[args.weld_partition]
        worker_indices = sorted(weld_indices_to_refine)[start : start + count]
        report["worker_partition"] = args.weld_partition
        report["worker_partitions"] = args.weld_partitions
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
            labels[index] for index in fixture_indices_to_refine
        }:
            raise FixtureTransitTourError("resume requires every selected transit exactly once")
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
        if args.probe_transit_results:
            checked = json.loads(args.probe_transit_results.read_text())
            new_results = {item["stage"]: item for item in checked["transits"]}
            if (
                checked["tour"] != tour
                or len(new_results) != len(checked["transits"])
                or set(new_results) != {labels[index] for index in probe_indices_to_refine}
            ):
                raise FixtureTransitTourError(
                    "probe transit results do not cover each selected transit once"
                )
            prior.update(new_results)
        if args.weld_transit_results:
            checked = json.loads(args.weld_transit_results.read_text())
            new_results = {item["stage"]: item for item in checked["transits"]}
            if (
                checked["tour"] != tour
                or len(new_results) != len(checked["transits"])
                or set(new_results) != {labels[index] for index in weld_indices_to_refine}
            ):
                raise FixtureTransitTourError(
                    "weld transit results do not cover each selected transit once"
                )
            prior.update(new_results)
        if args.reuse_modeled_outer:
            outer_saved = json.loads(args.reuse_modeled_outer.read_text())
            outer_labels = {
                labels[i]
                for i, (left, right, _) in neighbors.items()
                if left is None or right is None
            }
            outer_results = {
                item["stage"]: item
                for item in outer_saved["transits"]
                if item["stage"] in outer_labels
            }
            if (
                outer_saved["tour"] != tour
                or set(outer_results) != outer_labels
                or any(
                    item["status"] != "sampled_transit_fixture_join_checked"
                    for item in outer_results.values()
                )
            ):
                raise FixtureTransitTourError("checked modeled outer transit results required")
            prior.update(outer_results)
        if any(
            stage not in prior or prior[stage]["status"] != "failed" for stage in args.retry_stage
        ):
            raise FixtureTransitTourError("only previously failed transits may be retried")
    elif args.retry_stage:
        raise FixtureTransitTourError("retry requires a checkpoint")
    refinement_started = perf_counter()
    for index in worker_indices if worker_indices is not None else sorted(indices_to_refine):
        label = labels[index]
        row = rows[index]
        timing = clock["stages"][index]
        result = {"stage": label, "status": "in_progress"}
        transit_wall = perf_counter()
        transit_cpu = process_time()
        try:
            left, right, events = neighbors[index]
            if (
                label in prior
                and label not in args.retry_stage
                and (index not in probe_indices_to_refine or args.probe_transit_results)
                and (index not in weld_indices_to_refine or args.weld_transit_results)
            ):
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
                    if left is not None and left.label in checked_probe_labels:
                        resume_checks.extend(j._asdict() for j in diagnostic_joins([left, stage]))
                    if right is not None and right.label in checked_probe_labels:
                        resume_checks.extend(j._asdict() for j in diagnostic_joins([stage, right]))
                    if left is not None and left.label in checked_weld_labels:
                        resume_checks.extend(j._asdict() for j in diagnostic_joins([left, stage]))
                    if right is not None and right.label in checked_weld_labels:
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
            outer = left is None or right is None
            if outer and not args.free_outer_boundary:
                raise FixtureTransitTourError("unknown outer physical boundary velocity")
            begin = left.evaluate(left.duration) if left is not None else None
            finish = right.evaluate(Seconds(0.0)) if right is not None else None
            if (begin is not None and begin.velocity is None) or (
                finish is not None and finish.velocity is None
            ):
                raise FixtureTransitTourError("neighboring modeled velocity unavailable")
            v0 = None if begin is None else np.asarray(begin.velocity)
            v1 = None if finish is None else np.asarray(finish.velocity)
            if v0 is not None and v1 is not None:
                require_fixed_fixture_boundary_velocities(v0, v1)
            else:
                fixed = v1 if v0 is None else v0
                require_fixed_fixture_boundary_velocities(fixed, fixed)
            left_fixture = left is not None and left.label.startswith("fixture:")
            right_fixture = right is not None and right.label.startswith("fixture:")
            left_probe = left is not None and left.label in checked_probe_labels
            right_probe = right is not None and right.label in checked_probe_labels
            left_weld = left is not None and left.label in checked_weld_labels
            right_weld = right is not None and right.label in checked_weld_labels
            a0 = (
                np.asarray(begin.acceleration)
                if begin is not None
                and (outer or left_fixture or left_probe or left_weld)
                and begin.acceleration is not None
                else None
            )
            a1 = (
                np.asarray(finish.acceleration)
                if finish is not None
                and (outer or right_fixture or right_probe or right_weld)
                and finish.acceleration is not None
                else None
            )
            if outer and (
                (begin is not None and a0 is None) or (finish is not None and a1 is None)
            ):
                raise FixtureTransitTourError(
                    "outer transit neighboring modeled acceleration unavailable"
                )
            if ((left_fixture or left_probe or left_weld) and a0 is None) or (
                (right_fixture or right_probe or right_weld) and a1 is None
            ):
                raise FixtureTransitTourError("actual checked boundary acceleration unavailable")

            def boundaries(_rows, _timing, _index, *, start_velocity, end_velocity):
                return start_velocity, end_velocity, events

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
            if label in collision_witnesses:
                if outer:
                    raise FixtureTransitTourError("outer-boundary collision witnesses unsupported")
                previous = prior[label]
                if "optimized_curve" not in previous:
                    raise FixtureTransitTourError("saved failed transit curve required")
                seed = previous["optimized_curve"]["nodes"]
                seed_accelerations = [
                    previous["optimized_curve"]["a0"],
                    previous["optimized_curve"]["a1"],
                ]
                result["collision_witness_times_s"] = collision_witnesses[label]
            result["position_witness_attempts"] = []
            for attempt in range(
                MAX_POSITION_WITNESS_ROUNDS if args.adaptive_position_witness else 1
            ):
                solver = refine_free_boundary_transits if outer else refine_transits
                options = (
                    {
                        "initial_trust_box_size": args.outer_trust_box,
                        "outer_speed_preference_per_s3": args.outer_speed_preference_per_s3,
                    }
                    if outer
                    else {}
                )
                if label in collision_witnesses:
                    options["collision_witness_times_s"] = collision_witnesses[label]
                solved = solver(
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
                    **options,
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
            if outer:
                velocity_limits = velocity_extrema(
                    curve, Seconds(0.0), Seconds(transit["duration_s"])
                )
                continuous_speed_excess = [
                    max(abs(extreme["minimum"]), abs(extreme["maximum"])) - speeds[k]
                    for k, extreme in enumerate(velocity_limits)
                ]
                result["actual_velocity_extrema"] = velocity_limits
                result["raw_velocity_extrema_excess"] = [
                    max(0.0, float(value)) for value in continuous_speed_excess
                ]
                if any(value > SQP_FEASIBILITY_BAND for value in continuous_speed_excess):
                    raise FixtureTransitTourError("actual velocity extrema exceed sourced speeds")
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
            if left_probe:
                checks += [j._asdict() for j in diagnostic_joins([left, stage])]
            if right_probe:
                checks += [j._asdict() for j in diagnostic_joins([stage, right])]
            if left_weld:
                checks += [j._asdict() for j in diagnostic_joins([left, stage])]
            if right_weld:
                checks += [j._asdict() for j in diagnostic_joins([stage, right])]
            if outer and left is not None and not left_fixture and not left_probe and not left_weld:
                checks += [j._asdict() for j in diagnostic_joins([left, stage])]
            if (
                outer
                and right is not None
                and not right_fixture
                and not right_probe
                and not right_weld
            ):
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
                    "process_boundary": "Checked modeled probe and fixture derivatives where supplied; otherwise generated joint-linear velocity and free acceleration. Physical derivatives unknown.",
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
        result["wall_s_contended"] = perf_counter() - transit_wall
        result["cpu_s"] = process_time() - transit_cpu
        report["transits"].append(result)
        save()
    if worker_indices is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["process_cpu_s"] = float(usage.ru_utime + usage.ru_stime)
        report["refinement_wall_s_contended"] = perf_counter() - refinement_started
        report["peak_rss_bytes"] = int(
            usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        )
        report["worker_pid"] = getpid()
        report["scene_restored"] = snapshot(scene) == original
        report["status"] = (
            "completed_with_failures"
            if any(t["status"] == "failed" for t in report["transits"])
            else "sampled_transit_partition"
        )
        save()
        if not report["scene_restored"]:
            raise FixtureTransitTourError("scene state changed")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "partition": report["worker_partition"],
                    "transits": len(report["transits"]),
                    "failed": sum(t["status"] == "failed" for t in report["transits"]),
                }
            ),
            flush=True,
        )
        return
    path, corners = assemble_actual_tour(rows, clock, replacements)
    joins = [item._asdict() for item in diagnostic_joins(path)]
    band = PARITY_ULPS * np.finfo(float).eps
    report["assembled_stage_count"] = len(path)
    report["source_generated_modeled_tour_s"] = sum(
        float(stage["duration_s"]) for stage in original_clock["stages"]
    )
    report["assembled_modeled_tour_s"] = sum(
        float(stage["duration_s"]) for stage in clock["stages"]
    )
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
    report["largest_modeled_join_gaps"] = {}
    for field in ("velocity_delta", "acceleration_delta"):
        for component, axes in (
            ("track", (0,)),
            ("arm", tuple(range(1, 7))),
            ("positioner", (7,)),
            ("tailstock", (8,)),
        ):
            known = [j for j in joins if j[field] is not None]
            ordered = sorted(
                known, key=lambda j: max(abs(j[field][axis]) for axis in axes), reverse=True
            )
            report["largest_modeled_join_gaps"][f"{field}_{component}"] = [
                {
                    "left": j["left"],
                    "right": j["right"],
                    "maximum_abs_native_units": max(abs(j[field][axis]) for axis in axes),
                    "raw_delta": j[field],
                }
                for j in ordered[:5]
            ]
    report["fixture_join_diagnostics"] = fixture_join_diagnostics
    report["checked_neighbor_join_count"] = len(fixture_join_diagnostics)
    report["unchanged_generated_stages"] = (
        "original independently sampled path; not rechecked at 1025 samples in this experiment"
    )
    report["status"] = (
        "partial_tour_with_failed_transits_and_unresolved_process"
        if any(t["status"] == "failed" for t in report["transits"])
        or (args.weld_batch and report["checked_weld_batch"]["unresolved_groups"])
        else "fixture_transits_checked_process_continuity_unresolved"
    )
    if args.verify_partitions is not None:
        ranges = split_ranges(0, len(path), args.verify_partitions)
        if args.verify_partition < 0 or args.verify_partition >= len(ranges):
            raise FixtureTransitTourError("tour verification partition outside range")
        start, count = ranges[args.verify_partition]
        report["verification_partition"] = args.verify_partition
        report["verification_worker_pid"] = getpid()
        report["verification_start"] = start
        report["verification_count"] = count
        report["verification_scope"] = (
            "Independent finite actual-curve samples; process acceptance and physical timing unresolved."
        )
        report["verification_stages"] = []
        progress = args.output / "verification-progress.jsonl"
        for stage in path[start : start + count]:
            entry = {"stage": stage.label, "status": "in_progress"}
            try:
                if isinstance(stage, StationaryEvent):
                    q = np.asarray([stage.position])
                    entry["stationary_event"] = True
                    entry["undefined_velocity_samples"] = 0
                else:
                    ts = np.linspace(0.0, float(stage.duration), SAMPLES)
                    states = [stage.evaluate(Seconds(float(t))) for t in ts]
                    q = np.asarray([item.position for item in states])
                    velocities = [item.velocity[:7] for item in states if item.velocity is not None]
                    entry["undefined_velocity_samples"] = len(states) - len(velocities)
                    if velocities:
                        entry["sampled_speed_excess_native_units_s"] = np.maximum(
                            np.max(np.abs(np.asarray(velocities)), axis=0) - np.asarray(speeds), 0.0
                        ).tolist()
                entry["collision_and_position"] = inspect_sampled_curve(
                    checker, q, limits, SQP_FEASIBILITY_BAND
                )
                entry["status"] = (
                    "sampled_speed_excess"
                    if any(
                        value > SQP_FEASIBILITY_BAND
                        for value in entry.get("sampled_speed_excess_native_units_s", [])
                    )
                    else "sampled_checked"
                )
            except Exception as exc:
                entry.update(
                    {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
                )
            report["verification_stages"].append(entry)
            with progress.open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        report["verification_status"] = (
            "completed_with_failures"
            if any(s["status"] != "sampled_checked" for s in report["verification_stages"])
            else "sampled_partition_checked"
        )
        report["verification_scene_restored"] = snapshot(scene) == original
        if not report["verification_scene_restored"]:
            raise FixtureTransitTourError("verification scene state changed")
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
