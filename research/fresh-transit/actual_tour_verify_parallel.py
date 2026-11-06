"""Independently sample an assembled actual tour using process-owned scenes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from probe_search_group_parallel import split_ranges


class ActualTourVerifyError(RuntimeError):
    """Verification partitions or their independently checked output differ."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "resume-from",
        "checked-neighborhood",
        "probe-batch",
        "probe-transit-results",
        "reuse-modeled-outer",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fixture-duration-factor", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    recovery = json.loads(args.recovery.read_text())
    labels = [row["stage"] for row in recovery["verified"]["expanded"]["segments"]]
    if not labels or len(set(labels)) != len(labels):
        raise ActualTourVerifyError("unique generated stages required")
    ranges = split_ranges(0, len(labels), args.workers)
    if args.output.exists():
        raise ActualTourVerifyError("output already exists")
    args.output.mkdir(parents=True)
    common: list[str] = []
    for name in (
        "station",
        "tour_index",
        "recovery",
        "resume_from",
        "checked_neighborhood",
        "probe_batch",
        "probe_transit_results",
        "reuse_modeled_outer",
        "ifopt_library",
        "sqp_library",
    ):
        common.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    common.extend(
        ("--fixture-duration-factor", str(args.fixture_duration_factor), "--free-outer-boundary")
    )
    script = Path(__file__).with_name("fixture_transit_tour_trial.py")
    start_wall = perf_counter()
    children = []
    for worker, (start, count) in enumerate(ranges):
        directory = args.output / f"worker-{worker:02d}"
        log = (args.output / f"worker-{worker:02d}.log").open("w")
        command = [
            sys.executable,
            str(script),
            *common,
            "--verify-partitions",
            str(args.workers),
            "--verify-partition",
            str(worker),
            "--output",
            str(directory),
        ]
        children.append(
            (
                worker,
                start,
                count,
                directory,
                log,
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT),
            )
        )
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, int | str]] = []
    for worker, start, count, directory, log, process in children:
        code = process.wait()
        log.close()
        if code != 0 or not (directory / "results.json").exists():
            failures.append({"worker": worker, "exit_code": code})
            continue
        value = json.loads((directory / "results.json").read_text())
        if (
            not value["verification_scene_restored"]
            or value["verification_start"] != start
            or value["verification_count"] != count
            or [stage["stage"] for stage in value["verification_stages"]]
            != labels[start : start + count]
        ):
            failures.append({"worker": worker, "error": "scene, coverage or stage order differs"})
            continue
        outputs.append(value)
    elapsed = perf_counter() - start_wall
    if failures:
        (args.output / "worker-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise ActualTourVerifyError(
            f"{len(failures)} verifier workers failed; inspect logs and partial checkpoints"
        )
    if len({value["verification_worker_pid"] for value in outputs}) != len(outputs):
        raise ActualTourVerifyError("verifier processes are not distinct")
    stages = [
        stage
        for output in sorted(outputs, key=lambda x: x["verification_start"])
        for stage in output["verification_stages"]
    ]
    if [stage["stage"] for stage in stages] != labels:
        raise ActualTourVerifyError("complete ordered stage coverage required")
    failed = [stage for stage in stages if stage["status"] != "sampled_checked"]
    result = {
        "tour": int(recovery["verified"]["tour"]),
        "status": "completed_with_failures" if failed else "sampled_tour_checked",
        "scope": "Independent actual-curve sampled contacts, joint bounds and sampled sourced speeds; process geometry, continuous clearance, controller timing and physical boundaries remain unresolved.",
        "workers": [
            {
                "partition": x["verification_partition"],
                "pid": x["verification_worker_pid"],
                "start": x["verification_start"],
                "count": x["verification_count"],
                "scene_restored": x["verification_scene_restored"],
            }
            for x in outputs
        ],
        "wall_s_contended": elapsed,
        "stages": stages,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "stages": len(stages),
                "failed": len(failed),
                "wall_s_contended": elapsed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
