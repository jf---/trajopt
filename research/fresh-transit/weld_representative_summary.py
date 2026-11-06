"""Summarize explicitly selected generated block checks without solver traces."""

import argparse
import json
from pathlib import Path


class RepresentativeSummaryError(RuntimeError):
    """The requested generated-block evidence is incomplete or ambiguous."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first-batch", "endpoint-retry", "contact-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RepresentativeSummaryError("output already exists")
    first = json.loads(args.first_batch.read_text())
    retry = json.loads(args.endpoint_retry.read_text())
    audit = json.loads(args.contact_audit.read_text())
    selected = [
        entry
        for entry in first["representative_blocks"]
        if entry["status"] == "evaluated_process_acceptance_unresolved"
    ]
    selected.extend(retry["representative_blocks"])
    if len(selected) != 3 or len({entry["source_segment"] for entry in selected}) != 3:
        raise RepresentativeSummaryError("exactly three distinct completed blocks required")
    profiles = {
        (item["initial_trust_box_native_units"], item["process_weight"])
        for entry in selected
        for item in entry["attempts"]
    }
    if len(profiles) != 1:
        raise RepresentativeSummaryError("one common solver setting is required")
    trust_box, process_weight = profiles.pop()
    contact_coverage = {
        item["stage"]: item
        for block in audit["representative_blocks"]
        for item in block["contact_coverage"]
    }
    blocks = []
    for block in selected:
        final = block["attempts"][-1]
        if len(block["generated_process"]) != len(final["stages"]):
            raise RepresentativeSummaryError("generated and final stages differ")
        stages = []
        for generated, candidate in zip(block["generated_process"], final["stages"], strict=True):
            if generated["stage"] != candidate["stage"]:
                raise RepresentativeSummaryError("generated and final stage order differs")
            stages.append(
                {
                    "stage": generated["stage"],
                    "kind": generated["kind"],
                    "nominal_feed_m_s": generated["nominal_feed_m_s"],
                    "generated_target_error_m": generated["blend_target_position_error_m"],
                    "generated_feed_range_m_s": generated["modeled_feed_range_m_s"],
                    "generated_collision_and_position": generated["collision_and_position"],
                    "final_target_error_m": candidate["max_target_position_error_m"],
                    "final_orientation_error_rad": candidate["max_source_orientation_error_rad"],
                    "final_local_backward_m": candidate["target_progress_3d"][
                        "actual_local_tangent_backward_total_m"
                    ],
                    "final_feed_range_m_s": candidate["feed_range_m_s"],
                    "final_collision_and_position": candidate["collision"],
                    "final_separate_contact_coverage": contact_coverage.get(candidate["stage"]),
                    "final_position_extrema_excess": candidate["position_extrema_excess"],
                    "final_speed_extrema_excess": candidate["speed_extrema_excess"],
                }
            )
        blocks.append(
            {
                "source_segment": block["source_segment"],
                "stages": stages,
                "attempts": [
                    {
                        "solver_status": item["solver_status"],
                        "solver_s_contended": item["solver_s_contended"],
                    }
                    for item in block["attempts"]
                ],
                "collision_witnesses": block["collision_witnesses"],
                "initial_contact_episodes": [
                    {
                        "stage": item["stage"],
                        "penetrating_samples": item["penetrating_samples"],
                        "worst_contact": item["worst_contact"],
                    }
                    for item in block["contact_audits"]
                ],
                "generated_dynamics": block["generated_dynamics"],
                "final_dynamics": block["final_dynamics"],
            }
        )
    result = {
        "scope": "Historical provisional three-block runs: fixture state was not set before constructing the initial kinematic group, so initial-solve results may use stale fixture transforms. Retain failures; corrected-order reliability reruns are pending. Contended timing; no continuous clearance or process acceptance.",
        "initial_trust_box_native_units": trust_box,
        "process_weight": process_weight,
        "blocks": blocks,
        "initial_endpoint_failures": [
            {"source_segment": block["source_segment"], "error": block["error"]}
            for block in first["representative_blocks"]
            if block["status"] == "failed"
        ],
        "input_paths": {
            "first_batch": str(args.first_batch),
            "endpoint_retry": str(args.endpoint_retry),
            "contact_audit": str(args.contact_audit),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
