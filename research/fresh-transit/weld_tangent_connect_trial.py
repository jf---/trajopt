"""Regenerate both transit routes around the tangent-lead weld experiment."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage, weld_stage
from actual_stage_contract import Seconds, diagnostic_joins
from continuous_spline_jerk import build_jerk_model
from cubic_position_extrema import cubic_position_extrema
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveCollisionError, CurveLimitError, inspect_sampled_curve
from native_accel_transits import refine_transits
from ompl_only_transit import plan_ompl_transit
from process_curve_representation import build_process_curve
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from weld_only_inputs import FixtureState
from weld_tangent_connect import attach_fixture_to_route
from weld_tangent_leads import fit_lead

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters, Radians

# Same finite sampling as the earlier transit and process experiments.
SAMPLES = 1025
SPEED_BOUND_ULPS = 32


class WeldConnectError(RuntimeError):
    """The regenerated lead and transit cannot form the requested neighborhood."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "lead-evidence",
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
    lead_evidence = json.loads(args.lead_evidence.read_text())
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
    ):
        raise WeldConnectError("independently verified generated candidate required")
    if lead_evidence["status"] != "sampled_lead_bead_diagnostic_only":
        raise WeldConnectError("checked generated tangent leads required")
    if args.output.exists():
        raise WeldConnectError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    initial = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("tangent_lead_connect_nine", list(names)).getLimits().joint_limits
    )
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise WeldConnectError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    speeds = np.asarray(speeds)

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    rows, timing = candidate["generated_stages"], candidate["clock"]["stages"]
    index = next(i for i, row in enumerate(rows) if row["stage"] == args.weld_stage)
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "scope": "Two newly generated OMPL transits and actual fixed-clock C2 splines around one generated bead; no physical or manufacturing acceptance.",
        "routes": [],
        "stages": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        if (
            index < 2
            or index + 2 >= len(rows)
            or not all(rows[j]["stage"].startswith("weld:") for j in (index - 1, index, index + 1))
        ):
            raise WeldConnectError("single bead and adjacent leads required")
        before, after = rows[index - 2], rows[index + 2]
        if not before["stage"].startswith("transit:") or not after["stage"].startswith("transit:"):
            raise WeldConnectError("both adjacent transits required")
        bead = np.asarray(rows[index]["states"], dtype=float)
        fixture = bead[0, 7:]
        bead_time = float(timing[index]["duration_s"])
        bead_stage, bead_curve = build_process_curve(args.weld_stage, bead, bead_time)
        q_in = np.asarray(lead_evidence["generated_lead_samples"]["lead_in"])
        q_out = np.asarray(lead_evidence["generated_lead_samples"]["lead_out"])
        if not np.array_equal(q_in[-1], bead[0, :7]) or not np.array_equal(q_out[0], bead[-1, :7]):
            raise WeldConnectError("lead evidence belongs to another bead")
        tin, tout = float(timing[index - 1]["duration_s"]), float(timing[index + 1]["duration_s"])
        ti, to = np.linspace(0, tin, len(q_in)), np.linspace(0, tout, len(q_out))
        rawin, rawout = CubicSpline(ti, q_in, axis=0), CubicSpline(to, q_out, axis=0)
        curve_in = fit_lead(
            ti, q_in, (rawin(0, 1), rawin(0, 2)), (bead_curve(0, 1), bead_curve(0, 2))
        )
        curve_out = fit_lead(
            to,
            q_out,
            (bead_curve(bead_time, 1), bead_curve(bead_time, 2)),
            (rawout(tout, 1), rawout(tout, 2)),
        )
        lead_in = weld_stage(rows[index - 1]["stage"], tin, curve_in, 0.0, tuple(fixture))
        lead_out = weld_stage(rows[index + 1]["stage"], tout, curve_out, 0.0, tuple(fixture))
        start = np.asarray(before["states"][0], dtype=float)
        finish = np.asarray(after["states"][-1], dtype=float)
        new_first, new_last = np.r_[q_in[0], fixture], np.r_[q_out[-1], fixture]
        fixed = FixtureState(Radians(float(fixture[0])), Meters(float(fixture[1])))
        for label, a, b in (
            (before["stage"], start, new_first),
            (after["stage"], new_last, finish),
        ):
            route = plan_ompl_transit(env, info, names[:7], a[:7], b[:7], fixed)
            report["routes"].append(
                {
                    "stage": label,
                    "profile": route["profile"],
                    "ompl_s_contended": route["ompl_s"],
                    "sample_count": route["samples_checked"],
                    "states": np.asarray(route["states"]).tolist(),
                }
            )
            save()
        transit_before = attach_fixture_to_route(
            before["stage"], report["routes"][0]["states"], start, new_first
        )
        transit_after = attach_fixture_to_route(
            after["stage"], report["routes"][1]["states"], new_last, finish
        )
        process_values = np.vstack((q_in, bead[1:, :7], q_out[1:]))
        process_states = np.column_stack(
            (process_values, np.tile(fixture, (len(process_values), 1)))
        )
        process_row = {"stage": "generated_weld_group", "states": process_states.tolist()}
        mini = [transit_before, process_row, transit_after]
        mini_clock = {
            "stages": [
                timing[index - 2],
                {"stage": process_row["stage"], "duration_s": tin + bead_time + tout},
                timing[index + 2],
            ]
        }
        old_before = next(row for row in candidate["transits"] if row["stage"] == before["stage"])
        old_after = next(row for row in candidate["transits"] if row["stage"] == after["stage"])
        lead_start = lead_in.evaluate(Seconds(0))
        lead_end = lead_out.evaluate(lead_out.duration)
        outer_start = np.r_[old_before["v0"], 0.0, 0.0]
        outer_end = np.r_[old_after["v1"], 0.0, 0.0]

        def boundaries(_rows, _clock, i, *, start_velocity, end_velocity):
            if i == 0:
                return start_velocity, np.asarray(lead_start.velocity), ()
            if i == 2:
                return np.asarray(lead_end.velocity), end_velocity, ()
            raise WeldConnectError("unexpected transit index")

        def accelerations(i):
            if i == 0:
                return np.r_[old_before["a0"], 0.0, 0.0], np.asarray(lead_start.acceleration)
            if i == 2:
                return np.asarray(lead_end.acceleration), np.r_[old_after["a1"], 0.0, 0.0]
            raise WeldConnectError("unexpected transit index")

        begin = perf_counter()
        outcomes = refine_transits(
            env,
            info,
            names,
            mini,
            mini_clock,
            speeds,
            start_velocity=outer_start,
            end_velocity=outer_end,
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        report["solve_s_contended"] = perf_counter() - begin
        report["transit_curves"] = outcomes
        curves, actual_transits = [], []
        for out in outcomes:
            model = native.QuinticSplineModel.build(
                out["knots_s"],
                np.asarray(out["v0"]),
                np.asarray(out["a0"]),
                np.asarray(out["v1"]),
                np.asarray(out["a1"]),
            )
            nodes = np.asarray(out["nodes"])
            actual_transits.append(
                transit_stage(out["stage"], out["duration_s"], model, nodes, tuple(out["fixture"]))
            )
            curves.append(
                build_jerk_model(
                    out["knots_s"],
                    np.asarray(out["v0"]),
                    np.asarray(out["a0"]),
                    np.asarray(out["v1"]),
                    np.asarray(out["a1"]),
                ).curve(nodes)
            )
        stages = [actual_transits[0], lead_in, bead_stage, lead_out, actual_transits[1]]
        polynomials = [curves[0], curve_in, bead_curve, curve_out, curves[1]]
        report["joins"] = [entry._asdict() for entry in diagnostic_joins(stages)]
        failed = False
        for stage, polynomial in zip(stages, polynomials, strict=True):
            ts = np.linspace(0, stage.duration, SAMPLES)
            states = [stage.evaluate(Seconds(float(t))) for t in ts]
            q = np.asarray([x.position for x in states])
            vel = np.asarray([x.velocity for x in states])
            try:
                check = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            except (CurveCollisionError, CurveLimitError) as exc:
                check = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
                failed = True
            extrema = (
                cubic_position_extrema(polynomial)
                if stage is bead_stage
                else position_extrema(polynomial)
            )
            excess = [
                max(limits[i, 0] - x["minimum"], x["maximum"] - limits[i, 1], 0.0)
                for i, x in enumerate(extrema)
            ]
            ratio = np.max(np.abs(vel[:, :7]), axis=0) / speeds
            speed_pass = bool(np.all(ratio <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps))
            failed |= not speed_pass or any(x > SQP_FEASIBILITY_BAND for x in excess)
            result = {
                "stage": stage.label,
                "duration_s": float(stage.duration),
                "collision_and_position": check,
                "extrema_excess": excess,
                "sampled_speed_ratio": ratio.tolist(),
                "sampled_speeds_pass": speed_pass,
            }
            if stage in (lead_in, bead_stage, lead_out):
                poses = np.asarray([pose(item) for item in q])
                result["geometry"] = assess_process_geometry(poses[0], poses[-1], poses)
                feed = np.linalg.norm(np.gradient(poses[:, :3, 3], ts, axis=0), axis=1)
                result["modeled_feed_range_m_s"] = [float(feed.min()), float(feed.max())]
            report["stages"].append(result)
            save()
        report["status"] = (
            "failed_actual_curve_checks" if failed else "sampled_weld_neighborhood_diagnostic_only"
        )
        report["unresolved"] = [
            "outer physical boundary derivatives",
            "lead approach admissibility",
            "physical timing",
            "manufacturing tolerance",
            "continuous collision",
            "whole-tour checking",
        ]
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != initial:
        raise WeldConnectError("scene state changed")
    print(json.dumps({"status": report["status"], "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
