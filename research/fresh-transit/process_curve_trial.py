"""Compare a cubic fit with the verified generated process stages in one scene."""

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
from actual_stage_contract import diagnostic_joins
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from process_curve_check import inspect_process_curve
from process_curve_representation import build_process_curve
from runtime_libraries import require_loaded_libraries
from verified_linear_stages import actual_linear_stage

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters


class ProcessTrialError(RuntimeError):
    """The saved generated candidate does not satisfy the trial contract."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", action="append", required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    data = json.loads(args.input.read_text())
    result = data["result"]
    candidate = result["jerk"]
    if (
        result["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
    ):
        raise ProcessTrialError("previously verified generated candidate required")
    if args.output.exists():
        raise ProcessTrialError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    original = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, _, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    bounds = np.asarray(
        env.getJointGroup("fresh_process_fit", list(names)).getLimits().joint_limits
    )
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise ProcessTrialError(f"missing sourced velocity for {name}")
        speeds.append(float(entry["velocity"]))

    def pose_for(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    stages, timings = candidate["generated_stages"], candidate["clock"]["stages"]
    results = []
    for label in args.stage:
        begin = perf_counter()
        try:
            index = next(i for i, row in enumerate(stages) if row["stage"] == label)
            if not 0 < index < len(stages) - 1:
                raise ProcessTrialError("both neighboring stages required")
            q, duration = stages[index]["states"], float(timings[index]["duration_s"])
            report = inspect_process_curve(
                label, q, duration, checker, bounds, speeds, pose_for, SQP_FEASIBILITY_BAND
            )
            fitted, _ = build_process_curve(label, q, duration)
            neighbors = [
                actual_linear_stage(
                    stages[i]["stage"], stages[i]["states"], float(timings[i]["duration_s"])
                )
                for i in (index - 1, index + 1)
            ]
            report["neighbor_joins"] = [
                join._asdict() for join in diagnostic_joins([neighbors[0], fitted, neighbors[1]])
            ]
            report["status"] = (
                "sampled_curve_checks_completed_process_tolerance_unspecified"
                if report["sampled_sourced_speeds_pass"]
                else "failed_sampled_sourced_speed"
            )
        except Exception as exc:
            report = {
                "stage": label,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        report["elapsed_s_contended"] = perf_counter() - begin
        results.append(report)
        (args.output / "results.json").write_text(
            json.dumps({"scene_setup_s": setup, "stages": results}, indent=2, allow_nan=False)
            + "\n"
        )
        print(
            json.dumps(
                {
                    "stage": label,
                    "status": report["status"],
                    "elapsed_s": report["elapsed_s_contended"],
                }
            ),
            flush=True,
        )
    if snapshot(scene) != original:
        raise ProcessTrialError("scene state changed during process checks")


if __name__ == "__main__":
    main()
