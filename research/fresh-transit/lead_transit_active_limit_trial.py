"""Coordinate a generated lead and transit at an active joint limit."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage
from actual_stage_contract import (
    Acceleration9,
    ActualStage,
    KinematicState,
    Position9,
    Seconds,
    Velocity9,
    diagnostic_joins,
)
from continuous_spline_jerk import build_jerk_model
from fixture_composite_stage import fixture_composite_stage
from fixture_multichord import build_fixture_multichord
from fresh_prepare import prepare_fresh
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from master_analytic_ik import arm_ik
from native_accel_transits import refine_transits
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from rotation_targets import interpolate_rotations
from runtime_libraries import require_loaded_libraries
from shared_active_limit_velocity import shared_active_limit_velocity
from spline_velocity_extrema import velocity_extrema

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent finite curve checks.
PARITY_ULPS = 32768  # Binary64 same-boundary comparison, not physical tolerance.


class LeadTransitLimitError(RuntimeError):
    """The generated lead/transit neighborhood cannot be coordinated safely."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "recovery", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--transit", required=True)
    parser.add_argument("--fixture-duration-factor", type=float, required=True)
    parser.add_argument("--cartesian-track-coordination", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.fixture_duration_factor) or args.fixture_duration_factor <= 0:
        raise LeadTransitLimitError("positive common fixture duration factor required")
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise LeadTransitLimitError("verified fresh generated candidate required")
    if args.output.exists():
        raise LeadTransitLimitError("output already exists")
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
    labels = [r["stage"] for r in rows]
    index = labels.index(args.transit)
    if (
        index < 2
        or index + 1 >= len(rows)
        or not labels[index - 1].startswith("weld:")
        or not labels[index + 1].startswith("fixture:")
    ):
        raise LeadTransitLimitError(
            "selected transit needs a generated weld lead and following fixture"
        )
    lead_row, previous_row, transit_row = rows[index - 1], rows[index - 2], rows[index]
    lead_duration = float(clock["stages"][index - 1]["duration_s"])
    previous_duration = float(clock["stages"][index - 2]["duration_s"])
    fixture_end = index + 1
    while fixture_end < len(rows) and labels[fixture_end].startswith("fixture:"):
        fixture_end += 1
    fixture_rows = rows[index + 1 : fixture_end]
    fixture_durations = [
        args.fixture_duration_factor * float(t["duration_s"])
        for t in clock["stages"][index + 1 : fixture_end]
    ]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("lead_limit_trial_nine", list(names)).getLimits().joint_limits
    )
    report = {
        "status": "in_progress",
        "tour": tour,
        "scene_setup_s_contended": setup,
        "lead": lead_row["stage"],
        "transit": args.transit,
        "scope": "Generated lead and transit with shared active-limit derivatives; modeled fixture timing and physical boundaries remain unknown.",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        lead_q = np.asarray(lead_row["states"], dtype=float)
        lead_source = lead_q.copy()
        prev_q = np.asarray(previous_row["states"], dtype=float)
        if (
            lead_duration <= 0
            or previous_duration <= 0
            or not np.all(lead_q[:, 7:] == lead_q[0, 7:])
        ):
            raise LeadTransitLimitError("positive clocks and fixed fixture lead required")

        def pose(q):
            state = env.getState(dict(zip(names, map(float, q), strict=True)))
            return np.linalg.solve(
                np.asarray(state.link_transforms["workpiece_link"].matrix),
                np.asarray(state.link_transforms["nozzle_link"].matrix),
            )

        if args.cartesian_track_coordination:
            first_pose, last_pose = pose(lead_source[0]), pose(lead_source[-1])
            fixture_map = dict(zip(names[7:], map(float, lead_source[0, 7:]), strict=True))
            part = np.asarray(env.getState(fixture_map).link_transforms["workpiece_link"].matrix)
            rail = native.MonotoneFixtureAxis.build(
                [0.0, lead_duration], [float(lead_source[0, 0]), float(lead_source[-1, 0])]
            )
            reference = lead_source[0, :7].copy()
            generated = [reference.copy()]
            clear_counts = []
            for i in range(1, len(lead_source) - 1):
                fraction = i / (len(lead_source) - 1)
                target = np.eye(4)
                target[:3, 3] = (1.0 - fraction) * first_pose[:3, 3] + fraction * last_pose[:3, 3]
                target[:3, :3] = interpolate_rotations(
                    first_pose[:3, :3], last_pose[:3, :3], [fraction]
                )[0]
                track = float(rail.evaluate_many([fraction * lead_duration])[0, 0])
                solution = arm_ik(
                    env,
                    opw,
                    part @ target,
                    {**fixture_map, names[0]: track},
                    tcp_link="nozzle_link",
                    limits=np.asarray(limits7)[1:],
                )
                choices = [
                    np.r_[track, arm]
                    for arm in solution.solutions
                    if not any(
                        float(hit.distance_m) < 0
                        for hit in checker.contacts(np.r_[track, arm, lead_source[0, 7:]])
                    )
                ]
                if not choices:
                    raise LeadTransitLimitError(
                        f"generated Cartesian lead has no collision-clear IK at node {i}"
                    )
                clear_counts.append(len(choices))
                reference = min(
                    choices, key=lambda value: float(np.sum(((value - reference) / speeds) ** 2))
                )
                generated.append(reference.copy())
            generated.append(lead_source[-1, :7].copy())
            lead_q = np.column_stack(
                (np.asarray(generated), np.tile(lead_source[0, 7:], (len(generated), 1)))
            )
            if np.any(
                np.abs(np.diff(lead_q[:, :7], axis=0)) / (lead_duration / (len(lead_q) - 1))
                > np.asarray(speeds)
            ):
                raise LeadTransitLimitError(
                    "generated Cartesian lead has a joint branch step above sourced speed"
                )
            report["minimum_clear_ik_branches"] = min(clear_counts)
        start_v = (prev_q[-1, :7] - prev_q[-2, :7]) * (len(prev_q) - 1) / previous_duration
        original_end_v = (
            (lead_source[-1, :7] - lead_source[-2, :7]) * (len(lead_source) - 1) / lead_duration
        )
        candidate_end_v = (lead_q[-1, :7] - lead_q[-2, :7]) * (len(lead_q) - 1) / lead_duration
        shared_v = shared_active_limit_velocity(lead_q[-1, :7], candidate_end_v, limits[:7])
        # The generated predecessor is joint-linear; zero qdd is a modeled hypothesis.
        start_a = np.zeros(7)
        shared_a = np.zeros(7)
        times = np.linspace(0, lead_duration, len(lead_q))
        lead_model = native.QuinticSplineModel.build(
            times.tolist(), start_v, start_a, shared_v, shared_a
        )
        lead_curve = build_jerk_model(times, start_v, start_a, shared_v, shared_a).curve(
            lead_q[:, :7]
        )
        fixture = lead_q[0, 7:]

        def lead_evaluate(t):
            q, v, a = lead_model.evaluate(np.asarray([float(t)]), lead_q[:, :7], 2)
            return KinematicState(
                Position9(tuple(np.r_[q[0], fixture])),
                Velocity9(tuple(np.r_[v[0], 0.0, 0.0])),
                Acceleration9(tuple(np.r_[a[0], 0.0, 0.0])),
            )

        lead = ActualStage(lead_row["stage"], Seconds(lead_duration), lead_evaluate)
        f0, f1 = np.asarray(fixture_rows[0]["states"]), np.asarray(fixture_rows[-1]["states"])
        fv0 = (f0[-1] - f0[0]) / fixture_durations[0]
        fv1 = (f1[-1] - f1[0]) / fixture_durations[-1]
        fv0[7:] = fv1[7:] = 0.0
        fixture_model, fixture_nodes, stage_times = build_fixture_multichord(
            fixture_rows, fixture_durations, fv0, np.zeros(9), fv1, np.zeros(9)
        )
        targets = np.asarray(
            [fixture_rows[0]["states"][0], *[r["states"][-1] for r in fixture_rows]]
        )
        fixture_axes = tuple(
            native.MonotoneFixtureAxis.build(stage_times, targets[:, k].tolist()) for k in (7, 8)
        )
        following = fixture_composite_stage(
            fixture_rows[0]["stage"],
            fixture_model,
            fixture_nodes,
            fixture_axes,
            stage_times[0],
            stage_times[1],
        )
        right = following.evaluate(Seconds(0))
        if right.velocity is None or right.acceleration is None:
            raise LeadTransitLimitError("actual following fixture derivatives unavailable")

        def boundaries(_rows, _timing, _index, *, start_velocity, end_velocity):
            return np.asarray(start_velocity), np.asarray(end_velocity), []

        def accelerations(_index):
            return np.r_[shared_a, 0.0, 0.0], np.asarray(right.acceleration)

        solved = refine_transits(
            env,
            info,
            names,
            [transit_row],
            {"stages": [clock["stages"][index]]},
            speeds,
            start_velocity=np.r_[shared_v, 0.0, 0.0],
            end_velocity=np.asarray(right.velocity),
            boundary_resolver=boundaries,
            acceleration_resolver=accelerations,
        )
        if len(solved) != 1 or solved[0]["status"] != "SQPStatus.NLP_CONVERGED":
            raise LeadTransitLimitError("native transit solve did not converge")
        transit = solved[0]
        transit_model = native.QuinticSplineModel.build(
            transit["knots_s"],
            np.asarray(transit["v0"]),
            np.asarray(transit["a0"]),
            np.asarray(transit["v1"]),
            np.asarray(transit["a1"]),
        )
        transit_curve = build_jerk_model(
            transit["knots_s"],
            np.asarray(transit["v0"]),
            np.asarray(transit["a0"]),
            np.asarray(transit["v1"]),
            np.asarray(transit["a1"]),
        ).curve(np.asarray(transit["nodes"]))
        transit_stage_actual = transit_stage(
            args.transit, transit["duration_s"], transit_model, transit["nodes"], transit["fixture"]
        )
        joins = [j._asdict() for j in diagnostic_joins([lead, transit_stage_actual, following])]
        band = PARITY_ULPS * np.finfo(float).eps
        if any(
            j[k] is None or max(map(abs, j[k])) > band
            for j in joins
            for k in ("position_delta", "velocity_delta", "acceleration_delta")
        ):
            raise LeadTransitLimitError("lead, transit and fixture derivatives do not match")
        evaluations = []
        for stage, model, nodes, curve in (
            (lead, lead_model, lead_q[:, :7], lead_curve),
            (transit_stage_actual, transit_model, np.asarray(transit["nodes"]), transit_curve),
        ):
            ts = np.linspace(0, float(stage.duration), SAMPLES)
            q7, velocity = model.evaluate(ts, nodes, 1)
            q9 = np.column_stack((q7, np.tile(fixture, (len(q7), 1))))
            checked = inspect_sampled_curve(checker, q9, limits, SQP_FEASIBILITY_BAND)
            extrema = position_extrema(curve)
            excess = [
                max(limits[k, 0] - x["minimum"], x["maximum"] - limits[k, 1], 0.0)
                for k, x in enumerate(extrema)
            ]
            velocity_bounds = velocity_extrema(curve, 0.0, float(stage.duration))
            velocity_excess = [
                max(abs(x["minimum"]) - speeds[k], abs(x["maximum"]) - speeds[k], 0.0)
                for k, x in enumerate(velocity_bounds)
            ]
            if any(x > SQP_FEASIBILITY_BAND for x in [*excess, *velocity_excess]):
                raise LeadTransitLimitError(
                    f"{stage.label}: actual position/speed extrema exceed configured band"
                )
            entry = {
                "stage": stage.label,
                "collision_and_position": checked,
                "position_extrema": extrema,
                "velocity_extrema": velocity_bounds,
                "raw_position_excess": excess,
                "raw_velocity_excess": velocity_excess,
            }
            if stage.label == lead_row["stage"]:
                tool = np.asarray([pose(point) for point in q9])
                entry["generated_lead_geometry"] = assess_process_geometry(
                    pose(lead_source[0]), pose(lead_source[-1]), tool
                )
                feed = np.linalg.norm(np.gradient(tool[:, :3, 3], ts, axis=0), axis=1)
                entry["modeled_tool_feed_range_m_s"] = [float(np.min(feed)), float(np.max(feed))]
            evaluations.append(entry)
        report.update(
            {
                "status": "sampled_neighborhood_physical_and_process_acceptance_unresolved",
                "lead_model": {
                    "times_s": times.tolist(),
                    "nodes": lead_q[:, :7].tolist(),
                    "v0": start_v.tolist(),
                    "a0": start_a.tolist(),
                    "v1": shared_v.tolist(),
                    "a1": shared_a.tolist(),
                    "fixture": fixture.tolist(),
                },
                "original_generated_lead_end_velocity": original_end_v.tolist(),
                "candidate_generated_lead_end_velocity": candidate_end_v.tolist(),
                "shared_limit_velocity": shared_v.tolist(),
                "cartesian_track_coordination": args.cartesian_track_coordination,
                "fixture_duration_factor": args.fixture_duration_factor,
                "transit_curve": transit,
                "joins": joins,
                "evaluations": evaluations,
                "physical_boundaries": "unknown; predecessor secant and zero boundary accelerations are modeled hypotheses",
            }
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
        raise LeadTransitLimitError("scene state changed")
    print(json.dumps({"status": report["status"], "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
