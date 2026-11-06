"""Integrate checked generated SearchL candidates into generated probe groups."""

import argparse
import json
import resource
import sys
import traceback
from os import getpid
from pathlib import Path
from time import perf_counter, process_time

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage
from actual_stage_contract import Seconds, diagnostic_joins
from continuous_spline_jerk import build_jerk_model
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from probe_cruise_clock import search_cruise_clock
from probe_cruise_generate import generate_cruise_group
from probe_cruise_trial import one_sided_progress
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import interpolate_rotations, rotation_errors
from runtime_libraries import require_loaded_libraries
from search_candidate_adapter import checked_search_nodes
from spline_velocity_extrema import velocity_extrema
from weld_tangent_leads import fit_lead

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing actual-curve independent finite sampling policy.
PARITY_ULPS = 32768  # Binary64 boundary reconstruction, not process tolerance.


class ProbeSearchBatchError(RuntimeError):
    """Generated probe phases or checked SearchL candidates are incompatible."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "searches",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--probe-start", type=int, required=True)
    parser.add_argument("--probe-count", type=int, required=True)
    args = parser.parse_args()
    if args.probe_start < 0 or args.probe_count <= 0:
        raise ProbeSearchBatchError("nonnegative start and positive probe count required")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    search_results = json.loads((args.searches / "results.json").read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise ProbeSearchBatchError("verified fresh generated recovery required")
    if args.output.exists():
        raise ProbeSearchBatchError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    original = snapshot(scene)
    env, program, source, tours, names, opw, limits7, _, _, home, info, _ = scene["inputs"]
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
    labels = [row["stage"] for row in rows]
    indices = [i for i, label in enumerate(labels) if label.startswith("approach:")]
    if args.probe_start + args.probe_count > len(indices) or len(search_results["probes"]) != len(
        indices
    ):
        raise ProbeSearchBatchError("selected range or complete SearchL candidate count differs")
    limits = np.asarray(
        env.getJointGroup("probe_search_batch_nine", list(names)).getLimits().joint_limits
    )
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    report = {
        "tour": tour,
        "scene_setup_s_contended": setup,
        "probe_start": args.probe_start,
        "probe_count": args.probe_count,
        "scope": "Fresh generated probe phases and checked optimized SearchL, modeled clocks and assumed hits; no controller or manufacturing acceptance.",
        "groups": [],
    }
    (args.output / "checkpoint-metadata.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "groups"}, indent=2)
        + "\n"
    )

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    for probe_index in range(args.probe_start, args.probe_start + args.probe_count):
        group_wall = perf_counter()
        group_cpu = process_time()
        i = indices[probe_index]
        result = {"probe": probe_index, "status": "in_progress", "stages": []}
        try:
            phases, timing = rows[i : i + 4], clock["stages"][i : i + 4]
            generated, evidence = generate_cruise_group(
                env, opw, info, names, limits7, speeds, phases, timing
            )
            probe = search_results["probes"][probe_index]
            full = json.loads(
                (args.searches / f"probe-{probe_index:02d}" / "results.json").read_text()
            )
            search = evidence["process"][1]
            optimized = checked_search_nodes(search, probe, full, probe_index)
            search["ik_states"] = optimized.tolist()
            t = np.linspace(0.0, search["duration_s"], len(optimized))
            v0, a0, v1, a1 = (np.asarray(search[key]) for key in ("v0", "a0", "v1", "a1"))
            model = native.QuinticSplineModel.build(t.tolist(), v0, a0, v1, a1)
            fixture = tuple(np.asarray(phases[1]["states"])[0, 7:])
            generated[1] = transit_stage(
                search["stage"], search["duration_s"], model, optimized, fixture
            )
            joins = [j._asdict() for j in diagnostic_joins(generated)]
            band = PARITY_ULPS * np.finfo(float).eps
            if any(
                j[key] is None or max(map(abs, j[key])) > band
                for j in joins
                for key in ("position_delta", "velocity_delta", "acceleration_delta")
            ):
                raise ProbeSearchBatchError(
                    "generated and optimized probe boundary derivatives differ"
                )
            result["process_joins"] = joins
            result["process"] = evidence["process"]
            for j, stage in enumerate(generated):
                info_phase = evidence["process"][j]
                points = np.asarray(info_phase["ik_states"])
                stage_times = np.linspace(0.0, float(stage.duration), len(points))
                va = [np.asarray(info_phase[key]) for key in ("v0", "a0", "v1", "a1")]
                curve = (
                    build_jerk_model(stage_times, *va).curve(points)
                    if j == 1
                    else fit_lead(stage_times, points, (va[0], va[1]), (va[2], va[3]))
                )
                ts = np.linspace(0.0, float(stage.duration), SAMPLES)
                states = [stage.evaluate(Seconds(float(t))) for t in ts]
                q = np.asarray([s.position for s in states])
                checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
                extrema = position_extrema(curve)
                position_excess = [
                    max(limits[k, 0] - x["minimum"], x["maximum"] - limits[k, 1], 0.0)
                    for k, x in enumerate(extrema)
                ]
                speed_extrema = velocity_extrema(curve, 0.0, float(stage.duration))
                speed_excess = [
                    max(abs(x["minimum"]) - speeds[k], abs(x["maximum"]) - speeds[k], 0.0)
                    for k, x in enumerate(speed_extrema)
                ]
                if any(x > SQP_FEASIBILITY_BAND for x in [*position_excess, *speed_excess]):
                    raise ProbeSearchBatchError(
                        f"{stage.label}: actual position or sourced speed exceeds configured band"
                    )
                first, last = (
                    pose(np.asarray(phases[j]["states"])[0]),
                    pose(np.asarray(phases[j]["states"])[-1]),
                )
                tool = np.asarray([pose(point) for point in q])
                geometry = assess_process_geometry(first, last, tool)
                feed = np.linalg.norm(np.gradient(tool[:, :3, 3], ts, axis=0), axis=1)
                if j in (1, 2):
                    _, progress_clock = search_cruise_clock(float(info_phase["source_duration_s"]))
                    progress = progress_clock(ts)[0]
                else:
                    progress = one_sided_progress(
                        ts, float(stage.duration), 1.0 if j == 0 else 0.0, 1.0 if j == 3 else 0.0
                    )
                target_position = first[:3, 3] + progress[:, None] * (last[:3, 3] - first[:3, 3])
                target_rotation = interpolate_rotations(first[:3, :3], last[:3, :3], progress)
                result["stages"].append(
                    {
                        "stage": stage.label,
                        "duration_s": float(stage.duration),
                        "collision_and_position": checked,
                        "raw_position_extrema_excess": position_excess,
                        "raw_speed_extrema_excess": speed_excess,
                        "geometry": geometry,
                        "modeled_feed_range_m_s": [float(feed.min()), float(feed.max())],
                        "max_clock_position_error_m": float(
                            np.linalg.norm(tool[:, :3, 3] - target_position, axis=1).max()
                        ),
                        "max_clock_orientation_error_rad": float(
                            rotation_errors(target_rotation, tool[:, :3, :3]).max()
                        ),
                    }
                )
            result["status"] = "sampled_group_physical_and_process_acceptance_unresolved"
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        result["wall_s_contended"] = perf_counter() - group_wall
        result["cpu_s"] = process_time() - group_cpu
        report["groups"].append(result)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    report["status"] = (
        "completed_with_failures"
        if any(x["status"] == "failed" for x in report["groups"])
        else "sampled_groups_process_acceptance_unresolved"
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["process_cpu_s"] = float(usage.ru_utime + usage.ru_stime)
    report["peak_rss_bytes"] = int(
        usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    )
    report["worker_pid"] = getpid()
    report["scene_restored"] = snapshot(scene) == original
    save()
    if not report["scene_restored"]:
        raise ProbeSearchBatchError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "groups": len(report["groups"]),
                "failed": sum(g["status"] == "failed" for g in report["groups"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
