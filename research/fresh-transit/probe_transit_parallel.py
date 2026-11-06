"""Connect probe-adjacent transits in independent scene-owning processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from probe_search_group_parallel import split_ranges


class TransitParallelError(RuntimeError):
    """Probe transit partitions or their independently checked outputs differ."""


def expected_transits(recovery: dict[str, Any], batch: dict[str, Any]) -> list[str]:
    rows = recovery["verified"]["expanded"]["segments"]
    labels = [row["stage"] for row in rows]
    indices = {label: i for i, label in enumerate(labels)}
    selected: set[int] = set()
    for group in batch["groups"]:
        if group["status"] != "sampled_group_physical_and_process_acceptance_unresolved":
            raise TransitParallelError("probe group did not pass its actual-curve checks")
        phases = group["process"]
        if len(phases) != 4:
            raise TransitParallelError("four generated probe phases required")
        i = indices[phases[0]["stage"]]
        if (
            labels[i : i + 4] != [phase["stage"] for phase in phases]
            or i == 0
            or i + 4 >= len(rows)
        ):
            raise TransitParallelError("probe stages or adjacent transit indices differ")
        if not labels[i - 1].startswith("transit:") or not labels[i + 4].startswith("transit:"):
            raise TransitParallelError("probe group is not bounded by generated transits")
        selected.update((i - 1, i + 4))
    return [labels[i] for i in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "resume-from",
        "probe-batch",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fixture-duration-factor", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    recovery = json.loads(args.recovery.read_text())
    batch = json.loads(args.probe_batch.read_text())
    expected = expected_transits(recovery, batch)
    ranges = split_ranges(0, len(expected), args.workers)
    if args.output.exists():
        raise TransitParallelError("output already exists")
    args.output.mkdir(parents=True)
    script = Path(__file__).with_name("fixture_transit_tour_trial.py")
    common: list[str] = []
    for name in (
        "station",
        "tour_index",
        "recovery",
        "resume_from",
        "probe_batch",
        "ifopt_library",
        "sqp_library",
    ):
        common.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    common.extend(("--fixture-duration-factor", str(args.fixture_duration_factor)))
    began = perf_counter()
    children = []
    for worker, (start, count) in enumerate(ranges):
        directory = args.output / f"worker-{worker:02d}"
        log = (args.output / f"worker-{worker:02d}.log").open("w")
        command = [
            sys.executable,
            str(script),
            *common,
            "--probe-partitions",
            str(args.workers),
            "--probe-partition",
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
    outputs = []
    failures: list[dict[str, int | str]] = []
    for worker, start, count, directory, log, child in children:
        code = child.wait()
        log.close()
        if code != 0 or not (directory / "results.json").exists():
            failures.append({"worker": worker, "start": start, "count": count, "exit_code": code})
            continue
        result = json.loads((directory / "results.json").read_text())
        if (
            not result["scene_restored"]
            or [t["stage"] for t in result["transits"]] != expected[start : start + count]
        ):
            failures.append(
                {
                    "worker": worker,
                    "start": start,
                    "count": count,
                    "error": "scene or partition differs",
                }
            )
            continue
        outputs.append(result)
    elapsed = perf_counter() - began
    if failures:
        (args.output / "worker-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise TransitParallelError(f"{len(failures)} transit workers failed; inspect saved logs")
    if len({result["worker_pid"] for result in outputs}) != len(outputs):
        raise TransitParallelError("worker process ownership is not distinct")
    by_label = {t["stage"]: t for result in outputs for t in result["transits"]}
    if len(by_label) != len(expected) or set(by_label) != set(expected):
        raise TransitParallelError("transit worker outputs overlap or are incomplete")
    transits = [by_label[label] for label in expected]
    result = {
        "tour": int(recovery["verified"]["tour"]),
        "fixture_duration_factor": args.fixture_duration_factor,
        "status": "completed_with_failures"
        if any(t["status"] == "failed" for t in transits)
        else "sampled_transit_partitions",
        "work_wall_s_contended": elapsed,
        "workers": [
            {
                "partition": x["worker_partition"],
                "worker_pid": x["worker_pid"],
                "scene_restored": x["scene_restored"],
                "scene_setup_s_contended": x["scene_setup_s_contended"],
                "refinement_wall_s_contended": x["refinement_wall_s_contended"],
                "process_cpu_s": x["process_cpu_s"],
                "peak_rss_bytes": x["peak_rss_bytes"],
            }
            for x in outputs
        ],
        "transits": transits,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "workers": len(outputs),
                "transits": len(transits),
                "failed": sum(t["status"] == "failed" for t in transits),
                "work_wall_s_contended": elapsed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
