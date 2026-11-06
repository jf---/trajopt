"""Assemble and independently sample a fresh tour with two replaced neighborhoods."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
from actual_stage_adapters import transit_stage, weld_stage
from actual_stage_contract import ActualStage, Seconds, StationaryEvent, diagnostic_joins
from actual_tour_assembly import assemble_actual_tour
from continuous_spline_jerk import build_jerk_model
from cubic_position_extrema import cubic_position_extrema
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import CurveLimitError, inspect_sampled_curve
from probe_group_coordination import build_probe_group
from process_curve_representation import build_process_curve
from process_geometry import assess_process_geometry
from quintic_position_extrema import position_extrema
from runtime_libraries import require_loaded_libraries
from scipy.interpolate import CubicSpline
from weld_tangent_leads import fit_lead

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Existing independent spline sampling density.
SPEED_BOUND_ULPS = 32  # Binary64 allowance at an exactly declared speed bound.


class TourTrialError(RuntimeError):
    """The saved generated candidate or supplied replacements are incompatible."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "probe-batch",
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
    probe = json.loads(args.probe_batch.read_text())
    lead = json.loads(args.lead_evidence.read_text())
    weld = json.loads(args.weld_evidence.read_text())
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
        or probe["status"] != "optimization_completed_awaiting_actual_curve_verification"
        or weld["status"] != "sampled_weld_neighborhood_diagnostic_only"
        or lead["status"] != "sampled_lead_bead_diagnostic_only"
    ):
        raise TourTrialError(
            "verified generated input and completed diagnostic neighborhoods required"
        )
    if args.output.exists():
        raise TourTrialError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    initial = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, _, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("complete_tour_nine", list(names)).getLimits().joint_limits
    )
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise TourTrialError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    speeds = np.asarray(speeds)

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    rows, clock = candidate["generated_stages"], candidate["clock"]
    timings = clock["stages"]
    report = {
        "status": "in_progress",
        "scene_setup_s_contended": setup,
        "stages": [],
        "scope": "Fresh generated Tour with coordinated probes and one regenerated tangent-lead weld; finite sampling, modeled clocks; no whole-tour physical or manufacturing acceptance.",
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        wi = next(i for i, row in enumerate(rows) if row["stage"] == args.weld_stage)
        if wi < 2 or wi + 2 >= len(rows):
            raise TourTrialError("selected neighborhoods are incomplete")
        replacements = {}
        extent_curves = {}
        expected_groups = [
            row["stage"].split(":")[1] for row in rows if row["stage"].startswith("approach:")
        ]
        if [item["probe"] for item in probe["groups"]] != expected_groups:
            raise TourTrialError("probe batch does not cover every source probe in order")
        for group in probe["groups"]:
            pi = next(
                i for i, row in enumerate(rows) if row["stage"] == f"approach:{group['probe']}"
            )
            stages, probe_curves = build_probe_group(
                rows[pi : pi + 4], [float(t["duration_s"]) for t in timings[pi : pi + 4]]
            )
            replacements.update({stage.label: stage for stage in stages})
            extent_curves.update(
                {
                    stage.label: (curve, cubic_position_extrema)
                    for stage, curve in zip(stages, probe_curves, strict=True)
                }
            )
        bead = np.asarray(rows[wi]["states"], dtype=float)
        bead_time = float(timings[wi]["duration_s"])
        bead_stage, bead_curve = build_process_curve(args.weld_stage, bead, bead_time)
        qin = np.asarray(lead["generated_lead_samples"]["lead_in"])
        qout = np.asarray(lead["generated_lead_samples"]["lead_out"])
        if not np.array_equal(qin[-1], bead[0, :7]) or not np.array_equal(qout[0], bead[-1, :7]):
            raise TourTrialError("lead samples do not match bead endpoints")
        tin, tout = float(timings[wi - 1]["duration_s"]), float(timings[wi + 1]["duration_s"])
        ti, to = np.linspace(0, tin, len(qin)), np.linspace(0, tout, len(qout))
        rawin, rawout = CubicSpline(ti, qin, axis=0), CubicSpline(to, qout, axis=0)
        curvein = fit_lead(
            ti, qin, (rawin(0, 1), rawin(0, 2)), (bead_curve(0, 1), bead_curve(0, 2))
        )
        curveout = fit_lead(
            to,
            qout,
            (bead_curve(bead_time, 1), bead_curve(bead_time, 2)),
            (rawout(tout, 1), rawout(tout, 2)),
        )
        fixture = tuple(map(float, bead[0, 7:]))
        leads = [
            weld_stage(rows[wi - 1]["stage"], tin, curvein, 0.0, fixture),
            bead_stage,
            weld_stage(rows[wi + 1]["stage"], tout, curveout, 0.0, fixture),
        ]
        replacements.update({stage.label: stage for stage in leads})
        extent_curves.update(
            {
                leads[0].label: (curvein, position_extrema),
                bead_stage.label: (bead_curve, cubic_position_extrema),
                leads[2].label: (curveout, position_extrema),
            }
        )
        transit_rows = {row["stage"]: row for row in candidate["transits"]}
        transit_rows.update({row["stage"]: row for row in weld["transit_curves"]})
        transit_rows.update({row["stage"]: row for row in probe["transits"]})
        expected = {row["stage"] for row in rows if row["stage"].startswith("transit:")}
        if set(transit_rows) != expected:
            raise TourTrialError("every transit needs an actual curve")
        for label, row in transit_rows.items():
            model = native.QuinticSplineModel.build(
                row["knots_s"],
                np.asarray(row["v0"]),
                np.asarray(row["a0"]),
                np.asarray(row["v1"]),
                np.asarray(row["a1"]),
            )
            replacements[label] = transit_stage(
                label, row["duration_s"], model, row["nodes"], tuple(row["fixture"])
            )
            curve = build_jerk_model(
                row["knots_s"],
                np.asarray(row["v0"]),
                np.asarray(row["a0"]),
                np.asarray(row["v1"]),
                np.asarray(row["a1"]),
            ).curve(np.asarray(row["nodes"]))
            extent_curves[label] = (curve, position_extrema)
        path, corners = assemble_actual_tour(rows, clock, replacements)
        report["unresolved_joint_linear_corners"] = corners
        report["ordered_stationary_events"] = [
            stage.label for stage in path if isinstance(stage, StationaryEvent)
        ]
        report["modeled_joins"] = [join._asdict() for join in diagnostic_joins(path)]
        report["probe_stop_assumption"] = (
            "Three modeled internal stops per probe; not proven physically necessary. SearchL hit is hypothetical at the commanded limit."
        )
        report["outer_physical_derivatives"] = "unknown"
        report["modeled_duration_s"] = sum(
            float(stage.duration) for stage in path if isinstance(stage, ActualStage)
        )
        for stage in path:
            if isinstance(stage, StationaryEvent):
                continue
            ts = np.linspace(0, stage.duration, SAMPLES)
            states = [stage.evaluate(Seconds(float(t))) for t in ts]
            q = np.asarray([item.position for item in states])
            available_vel = np.asarray(
                [item.velocity[:7] for item in states if item.velocity is not None]
            )
            ratio = np.max(np.abs(available_vel), axis=0) / speeds
            checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
            if stage.label in extent_curves:
                curve, extreme = extent_curves[stage.label]
                extrema = extreme(curve)
                excess = [
                    max(limits[i, 0] - e["minimum"], e["maximum"] - limits[i, 1], 0.0)
                    for i, e in enumerate(extrema)
                ]
                if any(value > SQP_FEASIBILITY_BAND for value in excess):
                    raise CurveLimitError(f"{stage.label}: position extrema exceed configured band")
            else:
                # Linear interpolation cannot exceed its endpoints on an axis.
                source = next(row for row in rows if row["stage"] == stage.label)
                knots = np.asarray(source["states"], dtype=float)
                excess = np.max(
                    np.maximum(
                        np.maximum(limits[:7, 0] - knots[:, :7], knots[:, :7] - limits[:7, 1]), 0.0
                    ),
                    axis=0,
                ).tolist()
                if any(value > SQP_FEASIBILITY_BAND for value in excess):
                    raise CurveLimitError(f"{stage.label}: linear extrema exceed configured band")
            entry = {
                "stage": stage.label,
                "duration_s": float(stage.duration),
                "sampled_speed_ratio": ratio.tolist(),
                "sampled_speeds_pass": bool(
                    np.all(ratio <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps)
                ),
                "collision_and_position": checked,
                "raw_position_extrema_excess": excess,
            }
            if stage.label.startswith(("approach:", "search:", "retract:", "withdraw:", "weld:")):
                poses = np.asarray([pose(state) for state in q])
                entry["geometry"] = assess_process_geometry(poses[0], poses[-1], poses)
                feed = np.linalg.norm(np.gradient(poses[:, :3, 3], ts, axis=0), axis=1)
                entry["modeled_feed_range_m_s"] = [float(feed.min()), float(feed.max())]
                if stage.label not in (leads[0].label, leads[-1].label):
                    source = next(row for row in rows if row["stage"] == stage.label)
                    source_endpoints = np.asarray(source["states"], dtype=float)[[0, -1]]
                    endpoint_residual = np.max(np.abs(q[[0, -1]] - source_endpoints), axis=0)
                    entry["original_endpoint_raw_residual_by_axis"] = endpoint_residual.tolist()
                    if np.any(endpoint_residual > SQP_FEASIBILITY_BAND):
                        raise TourTrialError(
                            f"{stage.label}: process endpoints differ from generated source"
                        )
            report["stages"].append(entry)
            save()
        position_breach = any(
            any(abs(value) > SQP_FEASIBILITY_BAND for value in join["position_delta"])
            for join in report["modeled_joins"]
        )
        if position_breach:
            report["status"] = "failed_modeled_position_join"
        elif not all(x["sampled_speeds_pass"] for x in report["stages"]):
            report["status"] = "failed_sampled_sourced_speed"
        else:
            report["status"] = "sampled_tour_physical_and_process_acceptance_unresolved"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    save()
    if snapshot(scene) != initial:
        raise TourTrialError("scene changed")
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
