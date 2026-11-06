"""Run disjoint generated-probe batches with one independent scene per process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


class ProbeParallelError(RuntimeError):
    """Worker ranges or independently produced probe results are incomplete."""


def split_ranges(start: int, count: int, workers: int) -> list[tuple[int, int]]:
    """Partition contiguous probe IDs evenly without overlaps or empty workers."""
    if start < 0 or count <= 0 or workers <= 0 or workers > count:
        raise ProbeParallelError("valid start, count and nonempty workers required")
    size, extra = divmod(count, workers)
    cursor = start
    ranges = []
    for index in range(workers):
        amount = size + int(index < extra)
        ranges.append((cursor, amount))
        cursor += amount
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "searches",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--probe-start", type=int, required=True)
    parser.add_argument("--probe-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    ranges = split_ranges(args.probe_start, args.probe_count, args.workers)
    if args.output.exists():
        raise ProbeParallelError("output already exists")
    args.output.mkdir(parents=True)
    script = Path(__file__).with_name("probe_search_group_batch_trial.py")
    common: list[str] = []
    for name in ("station", "tour_index", "recovery", "searches", "ifopt_library", "sqp_library"):
        common.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    began = perf_counter()
    children = []
    for worker, (start, count) in enumerate(ranges):
        directory = args.output / f"worker-{worker:02d}"
        log = (args.output / f"worker-{worker:02d}.log").open("w")
        command = [
            sys.executable,
            str(script),
            *common,
            "--probe-start",
            str(start),
            "--probe-count",
            str(count),
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
        if not result["scene_restored"]:
            failures.append(
                {"worker": worker, "start": start, "count": count, "error": "worker scene changed"}
            )
            continue
        if [group["probe"] for group in result["groups"]] != list(range(start, start + count)):
            failures.append(
                {
                    "worker": worker,
                    "start": start,
                    "count": count,
                    "error": "probe output range differs",
                }
            )
            continue
        outputs.append(result)
    elapsed = perf_counter() - began
    if failures:
        (args.output / "worker-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise ProbeParallelError(f"{len(failures)} workers failed; inspect their saved logs")
    groups = sorted(
        (group for result in outputs for group in result["groups"]), key=lambda x: x["probe"]
    )
    if [group["probe"] for group in groups] != list(
        range(args.probe_start, args.probe_start + args.probe_count)
    ):
        raise ProbeParallelError("combined worker results are incomplete or duplicated")
    tours = {result["tour"] for result in outputs}
    if len(tours) != 1:
        raise ProbeParallelError("worker tours differ")
    if len({result["worker_pid"] for result in outputs}) != len(outputs):
        raise ProbeParallelError("worker processes are not distinct")
    report = {
        "tour": tours.pop(),
        "probe_start": args.probe_start,
        "probe_count": args.probe_count,
        "status": "completed_with_failures"
        if any(group["status"] == "failed" for group in groups)
        else "sampled_groups_process_acceptance_unresolved",
        "scope": "Independent process-owned scenes; actual process groups checked but not yet connected to a whole tour.",
        "work_wall_s_contended": elapsed,
        "workers": [
            {
                "probe_start": result["probe_start"],
                "probe_count": result["probe_count"],
                "scene_setup_s_contended": result["scene_setup_s_contended"],
                "process_cpu_s": result["process_cpu_s"],
                "peak_rss_bytes": result["peak_rss_bytes"],
                "worker_pid": result["worker_pid"],
                "scene_restored": result["scene_restored"],
            }
            for result in outputs
        ],
        "groups": groups,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "workers": len(outputs),
                "groups": len(groups),
                "work_wall_s_contended": elapsed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
