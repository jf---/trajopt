"""Connect checked weld groups with one isolated scene per worker process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from probe_search_group_parallel import split_ranges


class WeldTransitParallelError(RuntimeError):
    """Checked weld topology or isolated worker results are inconsistent."""


def expected_transits(recovery: dict[str, Any], batch: dict[str, Any]) -> list[str]:
    """Select all distinct neighboring transits of explicitly checked weld groups."""
    rows = recovery["verified"]["expanded"]["segments"]
    labels = [row["stage"] for row in rows]
    indices = {label: i for i, label in enumerate(labels)}
    selected: set[int] = set()
    for bead in batch["beads"]:
        if bead["status"] != "sampled_exploratory_weld_only":
            continue
        names = [stage["stage"] for stage in bead["stages"]]
        if not names or any(name not in indices for name in names):
            raise WeldTransitParallelError("checked weld stage absent from generated tour")
        first = indices[names[0]]
        end = first + len(names)
        if first == 0 or end >= len(labels) or labels[first:end] != names:
            raise WeldTransitParallelError(
                "checked weld stages are not an interior contiguous cluster"
            )
        if not labels[first - 1].startswith("transit:") or not labels[end].startswith("transit:"):
            raise WeldTransitParallelError("checked weld cluster lacks adjacent transits")
        selected.update((first - 1, end))
    return [labels[i] for i in sorted(selected)]


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
        "weld-batch",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fixture-duration-factor", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--free-outer-boundary", action="store_true")
    parser.add_argument("--outer-trust-box", type=float)
    parser.add_argument("--outer-speed-preference-per-s3", type=float, default=0.0)
    args = parser.parse_args()
    recovery = json.loads(args.recovery.read_text())
    batch = json.loads(args.weld_batch.read_text())
    expected = expected_transits(recovery, batch)
    ranges = split_ranges(0, len(expected), args.workers)
    if args.output.exists():
        raise WeldTransitParallelError("output already exists")
    args.output.mkdir(parents=True)
    script = Path(__file__).with_name("fixture_transit_tour_trial.py")
    common: list[str] = []
    for name in (
        "station",
        "tour_index",
        "recovery",
        "resume_from",
        "checked_neighborhood",
        "probe_batch",
        "probe_transit_results",
        "weld_batch",
        "ifopt_library",
        "sqp_library",
    ):
        common.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    common.extend(
        (
            "--fixture-duration-factor",
            str(args.fixture_duration_factor),
            "--outer-speed-preference-per-s3",
            str(args.outer_speed_preference_per_s3),
        )
    )
    if args.free_outer_boundary:
        common.append("--free-outer-boundary")
    if args.outer_trust_box is not None:
        common.extend(("--outer-trust-box", str(args.outer_trust_box)))
    started = perf_counter()
    children = []
    for worker, (start, count) in enumerate(ranges):
        directory = args.output / f"worker-{worker:02d}"
        log = (args.output / f"worker-{worker:02d}.log").open("w")
        command = [
            sys.executable,
            str(script),
            *common,
            "--weld-partitions",
            str(args.workers),
            "--weld-partition",
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
    elapsed = perf_counter() - started
    if failures:
        (args.output / "worker-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise WeldTransitParallelError(f"{len(failures)} workers failed; inspect saved logs")
    if len({result["worker_pid"] for result in outputs}) != len(outputs):
        raise WeldTransitParallelError("worker process ownership is not distinct")
    by_label = {item["stage"]: item for output in outputs for item in output["transits"]}
    if len(by_label) != len(expected) or set(by_label) != set(expected):
        raise WeldTransitParallelError("worker outputs overlap or are incomplete")
    transits = [by_label[label] for label in expected]
    result = {
        "tour": int(recovery["verified"]["tour"]),
        "fixture_duration_factor": args.fixture_duration_factor,
        "status": "completed_with_failures"
        if any(t["status"] == "failed" for t in transits)
        else "sampled_transit_partitions",
        "wall_s_contended": elapsed,
        "workers": [
            {
                k: x[k]
                for k in (
                    "worker_partition",
                    "worker_pid",
                    "scene_restored",
                    "scene_setup_s_contended",
                    "refinement_wall_s_contended",
                    "process_cpu_s",
                    "peak_rss_bytes",
                )
            }
            for x in outputs
        ],
        "unresolved_weld_groups": [
            {"source_segment": b["source_segment"], "status": b["status"]}
            for b in batch["beads"]
            if b["status"] != "sampled_exploratory_weld_only"
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
                "wall_s_contended": elapsed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
