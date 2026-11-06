"""Replay one live generated-weld QP with matched PIQP cost-scaling settings."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _qp_scale_replay as replay
import numpy as np
from fresh_product_scene import initialize_scene, snapshot
from runtime_libraries import require_loaded_libraries
from tesseract_robotics import trajopt_sqp as sqp
from timed_weld_tour import preserve_scene
from weld_local_blend import build_local_weld_blend
from weld_piecewise_correction import correct_piecewise_weld

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

PROCESS_WEIGHT = 10.0  # Same shared process objective as the source experiment.
POSE_WEIGHT = 1.0  # Same timed Cartesian target objective.
TRUST_BOX = 0.01  # Same native mixed-axis trust-box policy.


class QpReplayError(RuntimeError):
    """The supplied generated inputs or the matched replay are incomplete."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "batch",
        "weld-source",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-segment", type=int, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise QpReplayError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    original = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    saved = json.loads(args.input.read_text())["result"]
    batch = json.loads(args.batch.read_text())
    source = json.loads(args.weld_source.read_text())
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or batch["status"] != "optimization_completed_awaiting_actual_curve_verification"
    ):
        raise QpReplayError("verified generated tour and connected batch required")
    matching = [item for item in batch["welds"] if item["source_segment"] == args.source_segment]
    original_matches = [
        item for item in source["beads"] if item["source_segment"] == args.source_segment
    ]
    if len(matching) != 1 or len(original_matches) != 1:
        raise QpReplayError("exactly one matching generated weld cluster required")
    bead, source_bead = matching[0], original_matches[0]
    labels = bead["stages"]
    if [stage["stage"] for stage in source_bead["stages"]] != labels or not np.array_equal(
        source_bead["fixture"], bead["fixture"]
    ):
        raise QpReplayError("generated weld stages or fixture differ")
    rows = {row["stage"]: row for row in saved["jerk"]["generated_stages"]}
    clocks = {row["stage"]: row for row in batch["clock"]["stages"]}
    durations = np.asarray([clocks[label]["duration_s"] for label in labels], dtype=float)
    generated = {
        **source_bead,
        "times_s": bead["times_s"],
        "joint_states_7": bead["joint_states_7"],
        "fixture": bead["fixture"],
    }
    limits = np.asarray(env.getJointGroup("qp_replay_nine", list(names)).getLimits().joint_limits)
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    if any(
        metadata[name]["speed_source"] not in ("datasheet", "cell_declared") for name in names[:7]
    ):
        raise QpReplayError("sourced arm and track speeds required")
    speeds = np.asarray([metadata[name]["velocity"] for name in names[:7]], dtype=float)

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    transforms = [pose(np.asarray(rows[labels[0]]["states"][0], dtype=float))]
    transforms.extend(pose(np.asarray(rows[label]["states"][-1], dtype=float)) for label in labels)
    points = np.asarray([item[:3, 3] for item in transforms])
    rotations = np.asarray([item[:3, :3] for item in transforms])
    target, orientation, blend = build_local_weld_blend(points, rotations, durations)
    report = {
        "status": "in_progress",
        "source_segment": args.source_segment,
        "scene_setup_s_contended": setup,
        "scope": "One live generated QP replayed twice; unchanged constraints, tolerances, weights and trust box. Not a whole-tour result.",
        "replays": [],
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    class Compare(sqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, problem, result):
            self.calls += 1
            report["callback_iteration"] = int(result.overall_iteration)
            report["replays"] = replay.compare(problem, result.best_var_vals, result.box_size)
            report["status"] = "matched_qp_replayed_stopping_callback"
            save()
            return False

    callback = Compare()
    with preserve_scene(env):
        result = correct_piecewise_weld(
            env,
            info,
            names,
            speeds,
            limits,
            checker,
            generated,
            [rows[label] for label in labels],
            durations,
            points,
            rotations,
            target,
            orientation,
            blend,
            pose,
            lambda result: None,
            process_weight=PROCESS_WEIGHT,
            initial_trust_box=TRUST_BOX,
            constrain_process_knots=True,
            generated_lead_order=True,
            source_orientation_override=orientation,
            coalesce_generated_samples=True,
            timed_pose_weight=POSE_WEIGHT,
            extra_callback=callback,
        )
    report["source_solver_status"] = result["solver_status"]
    report["callback_calls"] = callback.calls
    if callback.calls != 1 or len(report["replays"]) != 2:
        raise QpReplayError("one matched QP replay was not captured")
    if snapshot(scene) != original:
        raise QpReplayError("scene state changed")
    report["status"] = "matched_qp_replay_completed"
    save()
    print(json.dumps({"status": report["status"], "replays": report["replays"]}), flush=True)


if __name__ == "__main__":
    main()
