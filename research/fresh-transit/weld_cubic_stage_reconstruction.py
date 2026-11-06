"""Reconstruct a checked generated weld with stage-local native cubic maps."""

import argparse
import json
from pathlib import Path

import _quintic_spline as native
import numpy as np
from scipy.interpolate import CubicSpline

SAMPLES = 1025  # Same finite reporting grid used by the process checks.
ENDPOINT_ULPS = 256  # Binary64 clock arithmetic, not a process tolerance.


class WeldCubicReconstructionError(RuntimeError):
    """Generated cubic stages cannot be reconstructed at their source clock boundaries."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weld-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise WeldCubicReconstructionError("output already exists")
    batch = json.loads(args.weld_batch.read_text())
    groups = [b for b in batch["beads"] if b["status"] == "sampled_bead_order_unresolved"]
    if not groups:
        raise WeldCubicReconstructionError("no explicitly unresolved weld group")
    report = {
        "tour": batch["tour"],
        "scope": "Exact generated cubic reconstruction at modeled clock; no solve or physical acceptance.",
        "groups": [],
    }
    for bead in groups:
        original_times = np.asarray(bead["times_s"], dtype=float)
        original_nodes = np.asarray(bead["joint_states_7"], dtype=float)
        baseline = CubicSpline(original_times, original_nodes, axis=0)
        durations = np.asarray([s["duration_s"] for s in bead["stages"]], dtype=float)
        edges = np.r_[0.0, np.cumsum(durations)]
        output = {"source_segment": bead["source_segment"], "stages": [], "joins": []}
        stage_data = []
        for index, entry in enumerate(bead["stages"]):
            left, right = edges[index], edges[index + 1]
            band = ENDPOINT_ULPS * np.finfo(float).eps * max(1.0, abs(left), abs(right))
            knots = original_times[(original_times > left + band) & (original_times < right - band)]
            knots = np.r_[left, knots, right]
            model = native.CubicSplineStage.build(knots.tolist(), 7)
            variables = np.vstack([baseline(knots), baseline(left, 1), baseline(right, 1)])
            samples = np.linspace(left, right, SAMPLES)
            errors = []
            for order in range(4):
                # Stage-end jerk is one-sided; compare the appropriate side.
                eval_times = samples[:-1] if order == 3 and index + 1 < len(durations) else samples
                mapping = model.map(eval_times.tolist(), order)
                errors.append(
                    np.max(
                        np.abs(
                            mapping.basis @ variables + mapping.offset - baseline(eval_times, order)
                        ),
                        axis=0,
                    ).tolist()
                )
            jacobian_errors = []
            step = np.cbrt(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(variables))))
            for order in range(3):
                mapping = model.map(np.linspace(left, right, 9).tolist(), order)
                maximum = 0.0
                for column in range(len(variables)):
                    plus, minus = variables.copy(), variables.copy()
                    plus[column, 0] += step
                    minus[column, 0] -= step
                    numerical = (
                        (mapping.basis @ plus + mapping.offset)
                        - (mapping.basis @ minus + mapping.offset)
                    )[:, 0] / (2 * step)
                    maximum = max(
                        maximum, float(np.max(np.abs(numerical - mapping.basis[:, column])))
                    )
                jacobian_errors.append(maximum)
            stage_data.append((model, variables, float(left), float(right)))
            output["stages"].append(
                {
                    "stage": entry["stage"],
                    "modeled_duration_s": float(durations[index]),
                    "knots": len(knots),
                    "max_abs_reference_difference_by_axis": errors,
                    "jacobian_max_abs_fd_difference_by_order": jacobian_errors,
                }
            )
        for left, right in zip(stage_data[:-1], stage_data[1:], strict=True):
            deltas = []
            for order in range(3):
                lm, rm = left[0].map([left[3]], order), right[0].map([right[2]], order)
                deltas.append(
                    ((rm.basis @ right[1] + rm.offset) - (lm.basis @ left[1] + lm.offset))[
                        0
                    ].tolist()
                )
            output["joins"].append(
                {
                    "position_delta": deltas[0],
                    "velocity_delta": deltas[1],
                    "acceleration_delta": deltas[2],
                }
            )
        report["groups"].append(output)
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tour": report["tour"], "groups": len(report["groups"])}), flush=True)


if __name__ == "__main__":
    main()
