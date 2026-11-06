"""Connect all exploratory weld clusters to actual generated probe/fixture boundaries."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from actual_stage_adapters import weld_stage
from actual_stage_contract import Seconds
from event_aware_spline_boundaries import require_fixed_fixture_boundary_velocities
from fixture_easing_policy import ease_fixture_chains
from fresh_product_scene import initialize_scene, snapshot
from native_accel_transits import refine_transits
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from transit_neighbor_map import transit_neighbor_map
from verified_linear_stages import actual_linear_stage
from weld_tangent_leads import fit_lead


class WeldMultiConnectError(RuntimeError):
    """A generated weld cluster or its transit boundaries are incompatible."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "probe-batch",
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
    saved = json.loads(args.input.read_text())["result"]
    candidate = saved["jerk"]
    probes = json.loads(args.probe_batch.read_text())
    welds = json.loads(args.weld_evidence.read_text())
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
        or probes["status"] != "optimization_completed_awaiting_actual_curve_verification"
        or any(item["status"] != "sampled_exploratory_weld_only" for item in welds["beads"])
    ):
        raise WeldMultiConnectError(
            "verified fresh input, generated probes and checked exploratory welds required"
        )
    if args.output.exists():
        raise WeldMultiConnectError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    original = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise WeldMultiConnectError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    rows = [dict(row) for row in candidate["generated_stages"]]
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "groups": probes["groups"],
        "welds": [],
        "transits": [],
        "scope": "Generated probes and exploratory source-waypoint weld blends; newly refined shared transits await independent actual-curve verification.",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        clock, fixture_stages = ease_fixture_chains(rows, probes["clock"])
        timing = clock["stages"]
        labels = [row["stage"] for row in rows]
        actual = {}
        for group in probes["groups"]:
            for item in group["process"]:
                i = labels.index(item["stage"])
                nodes = np.asarray(item["ik_states"], dtype=float)
                duration = float(item["duration_s"])
                curve = fit_lead(
                    np.linspace(0, duration, len(nodes)),
                    nodes,
                    (np.asarray(item["v0"]), np.asarray(item["a0"])),
                    (np.asarray(item["v1"]), np.asarray(item["a1"])),
                )
                fixture = np.asarray(rows[i]["states"], dtype=float)[0, 7:]
                actual[i] = weld_stage(item["stage"], duration, curve, 0.0, tuple(fixture))
                rows[i] = {
                    "stage": item["stage"],
                    "states": np.column_stack((nodes, np.tile(fixture, (len(nodes), 1)))).tolist(),
                }
        affected = set()
        for bead in welds["beads"]:
            names_in_run = [stage["stage"] for stage in bead["stages"]]
            indices = [labels.index(label) for label in names_in_run]
            if (
                indices != list(range(indices[0], indices[0] + len(indices)))
                or indices[0] == 0
                or indices[-1] + 1 >= len(rows)
            ):
                raise WeldMultiConnectError("weld stages are not a complete contiguous cluster")
            if not rows[indices[0] - 1]["stage"].startswith("transit:") or not rows[
                indices[-1] + 1
            ]["stage"].startswith("transit:"):
                raise WeldMultiConnectError(
                    "weld cluster must have two generated neighboring transits"
                )
            times = np.asarray(bead["times_s"], dtype=float)
            nodes = np.asarray(bead["joint_states_7"], dtype=float).copy()
            # The matched-endpoint policy keeps the original generated outer states;
            # use those states exactly so both neighboring stages share their anchor.
            nodes[0] = np.asarray(rows[indices[0]]["states"], dtype=float)[0, :7]
            nodes[-1] = np.asarray(rows[indices[-1]]["states"], dtype=float)[-1, :7]
            fixture = np.asarray(bead["fixture"], dtype=float)
            curve = CubicSpline(times, nodes, axis=0)
            durations = np.asarray([float(timing[i]["duration_s"]) for i in indices])
            edges = np.r_[0.0, np.cumsum(durations)]
            for j, i in enumerate(indices):
                samples = curve(np.linspace(edges[j], edges[j + 1], len(rows[i]["states"])))
                if j == 0:
                    samples[0] = nodes[0]
                if j == len(indices) - 1:
                    samples[-1] = nodes[-1]
                actual[i] = weld_stage(
                    labels[i], float(durations[j]), curve, float(edges[j]), tuple(fixture)
                )
                rows[i] = {
                    "stage": labels[i],
                    "states": np.column_stack(
                        (samples, np.tile(fixture, (len(samples), 1)))
                    ).tolist(),
                }
            affected.update((indices[0] - 1, indices[-1] + 1))
            report["welds"].append(
                {
                    "source_segment": bead["source_segment"],
                    "stages": names_in_run,
                    "times_s": times.tolist(),
                    "joint_states_7": nodes.tolist(),
                    "fixture": fixture.tolist(),
                }
            )
        for i, row in enumerate(rows):
            if row["stage"].startswith("transit:") or i in actual:
                continue
            if row["stage"].startswith("fixture:"):
                if row["stage"] not in fixture_stages:
                    raise WeldMultiConnectError(f"{row['stage']}: fixture curve unavailable")
                actual[i] = fixture_stages[row["stage"]]
            else:
                actual[i] = actual_linear_stage(
                    row["stage"], row["states"], float(timing[i]["duration_s"])
                )
        neighbors = transit_neighbor_map(rows, timing, actual)
        previous = {row["stage"]: row for row in probes["transits"]}
        if set(previous) != {row["stage"] for row in rows if row["stage"].startswith("transit:")}:
            raise WeldMultiConnectError("probe batch must contain every transit")
        start = np.r_[previous[rows[0]["stage"]]["v0"], 0.0, 0.0]
        end = np.r_[previous[rows[-1]["stage"]]["v1"], 0.0, 0.0]
        astart = np.r_[previous[rows[0]["stage"]]["a0"], 0.0, 0.0]
        aend = np.r_[previous[rows[-1]["stage"]]["a1"], 0.0, 0.0]

        def boundaries(_rows, _timing, index, *, start_velocity, end_velocity):
            left, right, events = neighbors[index]
            v0 = start_velocity if left is None else left.evaluate(left.duration).velocity
            v1 = end_velocity if right is None else right.evaluate(Seconds(0)).velocity
            if v0 is None or v1 is None:
                raise WeldMultiConnectError(f"{labels[index]}: unknown neighboring velocity")
            require_fixed_fixture_boundary_velocities(v0, v1)
            return np.asarray(v0), np.asarray(v1), events

        def accelerations(index):
            left, right, _ = neighbors[index]
            a0 = astart if left is None else left.evaluate(left.duration).acceleration
            a1 = aend if right is None else right.evaluate(Seconds(0)).acceleration
            if a0 is None or a1 is None:
                raise WeldMultiConnectError(f"{labels[index]}: unknown neighboring acceleration")
            return np.asarray(a0), np.asarray(a1)

        selected = []
        selected_timing = []
        for i, (row, entry) in enumerate(zip(rows, timing, strict=True)):
            label = (
                row["stage"]
                if i in affected or not row["stage"].startswith("transit:")
                else "unchanged-" + row["stage"]
            )
            selected.append({**row, "stage": label})
            selected_timing.append({**entry, "stage": label})
        begin = perf_counter()
        solved = refine_transits(
            env,
            info,
            names,
            selected,
            {**clock, "stages": selected_timing},
            speeds,
            start_velocity=start,
            end_velocity=end,
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        report["solve_s_contended"] = perf_counter() - begin
        expected = {labels[i] for i in affected}
        if {row["stage"] for row in solved} != expected:
            raise WeldMultiConnectError("expected one refined transit for each affected boundary")
        previous.update({row["stage"]: row for row in solved})
        report["transits"] = [
            previous[row["stage"]] for row in rows if row["stage"].startswith("transit:")
        ]
        report["refined_transits"] = [labels[i] for i in sorted(affected)]
        report["clock"] = clock
        report["fixture_stages"] = list(fixture_stages)
        report["status"] = "optimization_completed_awaiting_actual_curve_verification"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != original:
        raise WeldMultiConnectError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "welds": len(report["welds"]),
                "transits": len(report["transits"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
