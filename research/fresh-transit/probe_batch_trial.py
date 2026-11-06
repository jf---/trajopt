"""Coordinate every generated probe and its shared transits in one scene."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from actual_stage_adapters import weld_stage
from actual_stage_contract import Seconds
from fresh_product_scene import initialize_scene, snapshot
from native_accel_transits import refine_transits
from probe_group_coordination import build_probe_group
from process_curve_representation import build_process_curve
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from weld_tangent_connect import attach_fixture_to_route
from weld_tangent_leads import fit_lead


class ProbeBatchError(RuntimeError):
    """The selected fresh tour cannot support this explicit probe ablation."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "lead-evidence",
        "weld-evidence",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--weld-stage", required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    saved = json.loads(args.input.read_text())["result"]
    candidate = saved["jerk"]
    lead = json.loads(args.lead_evidence.read_text())
    weld = json.loads(args.weld_evidence.read_text())
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
        or lead["status"] != "sampled_lead_bead_diagnostic_only"
        or weld["status"] != "sampled_weld_neighborhood_diagnostic_only"
    ):
        raise ProbeBatchError("verified fresh input and tangent weld evidence required")
    if args.output.exists():
        raise ProbeBatchError("output already exists")
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
            raise ProbeBatchError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    rows, timings = candidate["generated_stages"], candidate["clock"]["stages"]
    wi = next(i for i, row in enumerate(rows) if row["stage"] == args.weld_stage)
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "scope": "All generated probes with explicit modeled internal stops, shared transits and regenerated lead-in; not physical process acceptance.",
        "groups": [],
        "transits": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        if wi < 2 or not rows[wi - 2]["stage"].startswith("transit:"):
            raise ProbeBatchError("tangent lead must follow a generated transit")
        bead = np.asarray(rows[wi]["states"], dtype=float)
        bead_time = float(timings[wi]["duration_s"])
        _, bead_curve = build_process_curve(args.weld_stage, bead, bead_time)
        qin = np.asarray(lead["generated_lead_samples"]["lead_in"], dtype=float)
        fixture = bead[0, 7:]
        if not np.array_equal(qin[-1], bead[0, :7]):
            raise ProbeBatchError("tangent lead does not match bead")
        tin = float(timings[wi - 1]["duration_s"])
        times = np.linspace(0, tin, len(qin))
        raw = CubicSpline(times, qin, axis=0)
        lead_curve = fit_lead(
            times, qin, (raw(0, 1), raw(0, 2)), (bead_curve(0, 1), bead_curve(0, 2))
        )
        lead_stage = weld_stage(rows[wi - 1]["stage"], tin, lead_curve, 0.0, tuple(fixture))
        sources = [dict(row) for row in rows[:wi]]
        clock = [dict(row) for row in timings[:wi]]
        actual = {}
        for i, row in enumerate(sources):
            if not row["stage"].startswith("approach:"):
                continue
            if i + 4 > wi:
                raise ProbeBatchError("incomplete probe")
            group, _ = build_probe_group(
                sources[i : i + 4], [float(t["duration_s"]) for t in clock[i : i + 4]]
            )
            for j, stage in enumerate(group, start=i):
                actual[j] = stage
                clock[j]["duration_s"] = float(stage.duration)
            report["groups"].append(
                {"probe": row["stage"].split(":")[1], "stages": [stage.label for stage in group]}
            )
        if not report["groups"]:
            raise ProbeBatchError("no generated probes")
        actual[wi - 1] = lead_stage
        route = next(item for item in weld["routes"] if item["stage"] == sources[wi - 2]["stage"])
        new_start = np.asarray(sources[wi - 2]["states"][0], dtype=float)
        new_end = np.r_[qin[0], fixture]
        sources[wi - 2] = attach_fixture_to_route(
            sources[wi - 2]["stage"], route["states"], new_start, new_end
        )
        sources[wi - 1] = {
            "stage": lead_stage.label,
            "states": np.column_stack((qin, np.tile(fixture, (len(qin), 1)))).tolist(),
        }
        before = next(row for row in candidate["transits"] if row["stage"] == sources[0]["stage"])
        outer_v = np.r_[before["v0"], 0.0, 0.0]
        outer_a = np.r_[before["a0"], 0.0, 0.0]
        lead_first = lead_stage.evaluate(Seconds(0))

        def sides(index):
            left = None if index == 0 else actual.get(index - 1)
            right = actual.get(index + 1)
            if (index > 0 and left is None) or right is None:
                raise ProbeBatchError(f"{sources[index]['stage']}: unsupported neighboring stage")
            return left, right

        def boundaries(_rows, _clock, index, *, start_velocity, end_velocity):
            left, right = sides(index)
            return (
                start_velocity
                if left is None
                else np.asarray(left.evaluate(left.duration).velocity),
                np.asarray(right.evaluate(Seconds(0)).velocity),
                (),
            )

        def accelerations(index):
            left, right = sides(index)
            return (
                outer_a if left is None else np.asarray(left.evaluate(left.duration).acceleration),
                np.asarray(right.evaluate(Seconds(0)).acceleration),
            )

        begin = perf_counter()
        outcomes = refine_transits(
            env,
            info,
            names,
            sources,
            {"stages": clock},
            speeds,
            start_velocity=outer_v,
            end_velocity=np.asarray(lead_first.velocity),
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        report["solve_s_contended"] = perf_counter() - begin
        report["transits"] = outcomes
        report["status"] = "optimization_completed_awaiting_actual_curve_verification"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != original:
        raise ProbeBatchError("scene state changed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "groups": len(report["groups"]),
                "transits": len(report["transits"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
