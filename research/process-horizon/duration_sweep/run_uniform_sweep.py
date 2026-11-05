"""Replan and verify Tour09 timing policies using one retained scene."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter, process_time

from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene
from runtime_libraries import require_loaded_libraries
from solve_duration_policy import SQP_FEASIBILITY_BAND, solve_duration_policy
from verify_duration_policy import verify_duration_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=Path, required=True)
    parser.add_argument("--original-worktree", type=Path, required=True)
    parser.add_argument("--fresh-route", type=Path, required=True)
    parser.add_argument("--verified-route", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--fixture-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ifopt-library", type=Path, required=True)
    parser.add_argument("--sqp-library", type=Path, required=True)
    parser.add_argument("--factors", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--timing-note", default="CPU contention not characterized")
    args = parser.parse_args()
    if any(not math.isfinite(factor) or factor <= 0 for factor in args.factors) or len(
        set(args.factors)
    ) != len(args.factors):
        raise RuntimeError("duration factors must be positive, finite and unique")
    libraries = require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output directory must be empty to preserve previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    station = args.station.resolve(strict=True)
    original = args.original_worktree.resolve(strict=True)
    artifacts = args.artifacts.resolve(strict=True)
    fixture_policy = args.fixture_policy.resolve(strict=True)
    raw = json.loads((station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((station / "joint_speeds.json").read_text())["joint_speeds"]["joints"]
    started = perf_counter()
    cpu_started = process_time()
    scene = initialize_scene(station, original, output)
    setup_s = perf_counter() - started
    setup_cpu_s = process_time() - cpu_started
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
    segments, clock, speed = prepare_fresh(
        env,
        program,
        source,
        tours,
        home,
        names,
        raw,
        metadata,
        9,
        json.loads(args.fresh_route.read_text()),
        json.loads(args.verified_route.read_text()),
    )
    manifest = {
        "scope": "Fresh Tour09 generated inputs; uniform transit timing and actual-curve sampled verification. No controller or manufacturing acceptance.",
        "native_libraries": [str(path) for path in libraries],
        "scene_setup_s": setup_s,
        "scene_setup_cpu_s": setup_cpu_s,
        "timing_note": args.timing_note,
        "fixture_policy": str(fixture_policy),
        "policies": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for factor in args.factors:
        cpu_before_solve = process_time()
        policy = solve_duration_policy(
            factor, scene, segments, clock, speed, output, artifacts, fixture_policy
        )
        solve_cpu_s = process_time() - cpu_before_solve
        cpu_before_check = process_time()
        verified = verify_duration_policy(
            factor, scene, segments, clock, speed, output, artifacts, fixture_policy
        )
        check_cpu_s = process_time() - cpu_before_check
        if verified["negative_contacts"]:
            raise RuntimeError(
                f"{factor:g}: whole-tour sampled contacts detected; report preserved"
            )
        if max(verified["raw_limit_excess"]) > SQP_FEASIBILITY_BAND:
            raise RuntimeError(
                f"{factor:g}: whole-tour configured joint limit check failed; report preserved"
            )
        manifest["policies"].append(
            {
                "factor": factor,
                "solver_and_transit_checks_s": policy["elapsed_s"],
                "solver_and_transit_checks_cpu_s": solve_cpu_s,
                "whole_tour_checks_s": verified["elapsed_s"],
                "whole_tour_checks_cpu_s": check_cpu_s,
                "modeled_tour_duration_s": verified["modeled_tour_duration_s"],
                "stage_count": verified["stage_count"],
                "checked_samples": verified["checked_samples"],
                "negative_contacts": verified["negative_contacts"],
                "overall_accepted": False,
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["policies"][-1]), flush=True)


if __name__ == "__main__":
    main()
