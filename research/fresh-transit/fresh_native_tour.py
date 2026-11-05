"""Run one fresh tour and native time-aware transit refinement in one scene."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import run_multi_episode_shared_scene as planner
import verify_multi_episode_shared_scene as verifier
from check_transit_sqp_callback import serializable
from fresh_scene import initialize_scene, snapshot
from fresh_tour_pipeline import modeled_tour_end_secants, plan_and_refine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--tour", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("output already exists")
    args.output.mkdir(parents=True)
    setup = perf_counter()
    scene = initialize_scene(args.station, args.original, args.output)
    setup_s = perf_counter() - setup
    before = snapshot(scene)
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    started = perf_counter()
    try:
        result = plan_and_refine(
            scene,
            args.tour,
            planner.run,
            verifier.verify,
            raw,
            metadata,
            boundary_policy=modeled_tour_end_secants,
        )
    except Exception as exc:
        (args.output / "failure.json").write_text(
            json.dumps(
                {
                    "tour": args.tour,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_s": perf_counter() - started,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    if before != snapshot(scene):
        raise RuntimeError("scene state was not restored")
    result["scene_setup_s"] = setup_s
    result["elapsed_after_scene_s"] = perf_counter() - started
    (args.output / "summary.json").write_text(
        json.dumps(serializable(result), indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "tour": args.tour,
                "status": result["status"],
                "transits": len(result["transits"]),
                "scene_setup_s": setup_s,
                "elapsed_after_scene_s": result["elapsed_after_scene_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
