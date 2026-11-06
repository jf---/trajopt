"""Compare a generated cubic block and a collision-witness retry on one clock."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from weld_block_dynamics import block_dynamics

AXES = (
    "track_joint",
    "mh50_20ii_joint_s",
    "mh50_20ii_joint_l",
    "mh50_20ii_joint_u",
    "mh50_20ii_joint_r",
    "mh50_20ii_joint_b",
    "mh50_20ii_joint_t",
)


class WeldBlockComparisonError(RuntimeError):
    """Saved generated and refined blocks cannot be compared on the same clock."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generated", "retry", "audit", "joint-speeds", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise WeldBlockComparisonError("output already exists")
    generated = json.loads(args.generated.read_text())
    retry = json.loads(args.retry.read_text())
    audit = json.loads(args.audit.read_text())["contact_audit"]
    candidates = retry["block_setting_results"]
    if len(candidates) != 1:
        raise WeldBlockComparisonError("one explicit refinement result required")
    candidate = candidates[0]
    labels = [item["stage"] for item in candidate["stages"]]
    sources = [
        bead
        for bead in generated["beads"]
        if all(label in [s["stage"] for s in bead["stages"]] for label in labels)
    ]
    if len(sources) != 1:
        raise WeldBlockComparisonError("generated source stages do not match the block")
    source = sources[0]
    source_stages = {stage["stage"]: stage for stage in source["stages"]}
    baseline = CubicSpline(
        np.asarray(source["times_s"]), np.asarray(source["joint_states_7"]), axis=0
    )
    source_curves, candidate_curves = [], []
    for curve in candidate["stage_curves"]:
        knots = np.asarray(curve["knots_s"], dtype=float)
        source_curves.append(
            CubicSpline(
                knots,
                baseline(knots),
                axis=0,
                bc_type=((1, baseline(knots[0], 1)), (1, baseline(knots[-1], 1))),
            )
        )
        candidate_curves.append(
            CubicSpline(
                knots,
                np.asarray(curve["nodes"]),
                axis=0,
                bc_type=((1, curve["v0"]), (1, curve["v1"])),
            )
        )
    metadata = json.loads(args.joint_speeds.read_text())["joint_speeds"]["joints"]
    if any(metadata[axis]["speed_source"] not in ("datasheet", "cell_declared") for axis in AXES):
        raise WeldBlockComparisonError("sourced velocity scales are required")
    scales = np.asarray([metadata[axis]["velocity"] for axis in AXES], dtype=float)
    result = {
        "scope": "Generated-only three-stage modeled curve; finite contact and process samples; not physical timing or process acceptance.",
        "axes": list(AXES),
        "speed_scales": scales.tolist(),
        "stages": labels,
        "generated_dynamics": block_dynamics(source_curves, scales),
        "retry_dynamics": block_dynamics(candidate_curves, scales),
        "generated_process": [source_stages[label] for label in labels],
        "retry_process": candidate["stages"],
        "retry_solver_status": candidate["solver_status"],
        "retry_solver_s_contended": candidate["solver_s_contended"],
        "retry_collision_witnesses": retry["collision_refinement"],
        "retry_contact_summary": {
            "penetrating_samples": audit["penetrating_samples"],
            "worst_contact": audit["worst_contact"],
        },
        "input_paths": {
            "generated": str(args.generated),
            "retry": str(args.retry),
            "audit": str(args.audit),
            "joint_speeds": str(args.joint_speeds),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
