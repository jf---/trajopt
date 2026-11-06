"""Combine disjoint generated probe checkpoints without losing failures."""

import argparse
import json
from pathlib import Path
from typing import Any


class ProbeMergeError(RuntimeError):
    """Probe batches overlap, are incomplete, or refer to different tours."""


def merge_probe_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Return contiguous probe groups with every explicit failure preserved."""
    if not batches or len({batch["tour"] for batch in batches}) != 1:
        raise ProbeMergeError("one tour and at least one batch required")
    groups = []
    for batch in batches:
        start, count = int(batch["probe_start"]), int(batch["probe_count"])
        entries = batch["groups"]
        if count <= 0 or [entry["probe"] for entry in entries] != list(range(start, start + count)):
            raise ProbeMergeError("each batch must contain its complete ordered range")
        groups.extend(entries)
    groups.sort(key=lambda item: item["probe"])
    indices = [item["probe"] for item in groups]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ProbeMergeError("batch ranges overlap or have gaps")
    failed = [item["probe"] for item in groups if item["status"] == "failed"]
    return {
        "tour": batches[0]["tour"],
        "probe_start": indices[0],
        "probe_count": len(indices),
        "status": "completed_with_failures"
        if failed
        else "sampled_groups_process_acceptance_unresolved",
        "failed_probes": failed,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ProbeMergeError("output already exists")
    result = merge_probe_batches([json.loads(path.read_text()) for path in args.batch])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
