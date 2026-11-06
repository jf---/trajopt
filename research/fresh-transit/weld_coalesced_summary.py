"""Summarize corrected-fixture coalescing checks from explicit saved inputs."""

import argparse
import json
from pathlib import Path

import numpy as np


class WeldCoalescedSummaryError(RuntimeError):
    """Saved source and coalesced results do not support this comparison."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "preview", "prior", "contact-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--contact-source-segment", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise WeldCoalescedSummaryError("output already exists")
    batch = json.loads(args.batch.read_text())
    preview = json.loads(args.preview.read_text())
    prior = json.loads(args.prior.read_text())
    audit = json.loads(args.contact_audit.read_text())
    blocks = batch["representative_blocks"]
    if len(blocks) != 3 or any(len(block["attempts"]) != 1 for block in blocks):
        raise WeldCoalescedSummaryError("three completed single-attempt blocks are required")
    old = next(
        block
        for block in prior["representative_blocks"]
        if block["source_segment"] == args.contact_source_segment
    )["attempts"][-1]
    current = next(
        block for block in blocks if block["source_segment"] == args.contact_source_segment
    )["attempts"][-1]
    for a, b in zip(old["stage_curves"], current["stage_curves"], strict=True):
        for key in ("knots_s", "nodes", "v0", "v1"):
            if not np.array_equal(a[key], b[key]):
                raise WeldCoalescedSummaryError(
                    "separate contact scan cannot be reused for different curves"
                )
    coverage = audit["representative_blocks"][0]["contact_coverage"]
    result = {
        "scope": "Three generated local blocks, corrected fixture ordering, common solver/coalescing policy; contended times and finite samples only. Manufacturing and controller acceptance unresolved.",
        "profile": {
            "initial_trust_box_native_units": blocks[0]["attempts"][0][
                "initial_trust_box_native_units"
            ],
            "process_weight": blocks[0]["attempts"][0]["process_weight"],
            "maximum_collision_retries": 1,
        },
        "preview": {
            "removed_generated_samples_s": preview["representative_blocks"][0]["attempts"][0][
                "removed_generated_samples_s"
            ],
            "stages": preview["representative_blocks"][0]["attempts"][0]["stages"],
            "initial_reconstruction_max_delta_by_order": preview["representative_blocks"][0][
                "attempts"
            ][0]["initial_reconstruction_max_delta_by_order"],
        },
        "blocks": [],
        "input_paths": {
            "batch": str(args.batch),
            "preview": str(args.preview),
            "prior": str(args.prior),
            "contact_audit": str(args.contact_audit),
        },
    }
    for block in blocks:
        attempt = block["attempts"][-1]
        result["blocks"].append(
            {
                "source_segment": block["source_segment"],
                "stages": block["stages"],
                "removed_generated_samples_s": attempt["removed_generated_samples_s"],
                "solver_status": attempt["solver_status"],
                "solver_s_contended": attempt["solver_s_contended"],
                "initial_constraint_violations": attempt["initial"]["constraint_violations"],
                "first_qp": attempt["trace"][0]["qp"] if attempt["trace"] else None,
                "source_dynamics": block["generated_dynamics"],
                "final_dynamics": block["final_dynamics"],
                "source_process": block["generated_process"],
                "final_process": attempt["stages"],
                "outer_boundary_delta_from_generated": attempt[
                    "outer_boundary_delta_from_generated"
                ],
                "reused_contact_coverage": coverage
                if block["source_segment"] == args.contact_source_segment
                else [],
            }
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
