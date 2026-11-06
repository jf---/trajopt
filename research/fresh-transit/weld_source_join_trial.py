"""Classify source bead and generated-lead Cartesian joins in one scene."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _quintic_spline  # noqa: F401  # Load selected native research libraries.
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401  # Verify selected SQP runtime.
from fresh_product_scene import initialize_scene, snapshot
from probe_join_geometry import inspect_probe_join
from probe_process_inputs import extract_process_tour
from probe_source_join_trial import pose_matrix
from runtime_libraries import require_loaded_libraries
from weld_only_inputs import WeldBead, extract_tour


class WeldSourceJoinError(RuntimeError):
    """The generated weld cluster cannot be matched to the source bead."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    saved = json.loads(args.input.read_text())["result"]
    candidate = saved["jerk"]
    if (
        saved["product"]["status"] != "independently_sampled_tour"
        or candidate["process_verification"]["status"] != "independently_sampled_tour"
    ):
        raise WeldSourceJoinError("verified fresh generated source required")
    if args.output.exists():
        raise WeldSourceJoinError("output already exists")
    args.output.mkdir(parents=True)
    begin = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - begin
    before = snapshot(scene)
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
    tour = int(candidate["tour"])
    task = extract_tour(source, tours, tour, "workpiece_link", home=home)
    process = extract_process_tour(program, task, probe_metadata_frame="workpiece_link")
    beads = [item for item in process.operations if isinstance(item, WeldBead)]
    runs = []
    for row in candidate["generated_stages"]:
        if row["stage"].startswith("weld:"):
            if not runs or runs[-1] is None:
                runs.append([])
            runs[-1].append(row)
        elif runs and runs[-1] is not None:
            runs.append(None)
    runs = [run for run in runs if run is not None]
    if len(runs) != len(beads):
        raise WeldSourceJoinError("generated weld clusters do not match source beads")

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    output = []
    for bead, run in zip(beads, runs, strict=True):
        poses = [pose_matrix(item) for item in bead.poses]
        if len(poses) < 2 or len(run) != len(poses) + 1:
            raise WeldSourceJoinError(
                f"weld {bead.segment}: source poses and generated edges differ"
            )
        generated_start = pose(run[0]["states"][0])
        generated_end = pose(run[-1]["states"][-1])
        joins = [
            {
                "kind": "generated_lead_in_to_source_bead",
                "left": run[0]["stage"],
                "right": run[1]["stage"],
                **inspect_probe_join(generated_start, poses[0], poses[1])._asdict(),
            }
        ]
        for i in range(1, len(poses) - 1):
            joins.append(
                {
                    "kind": "source_bead_internal",
                    "left": run[i]["stage"],
                    "right": run[i + 1]["stage"],
                    **inspect_probe_join(poses[i - 1], poses[i], poses[i + 1])._asdict(),
                }
            )
        joins.append(
            {
                "kind": "source_bead_to_generated_lead_out",
                "left": run[-2]["stage"],
                "right": run[-1]["stage"],
                **inspect_probe_join(poses[-2], poses[-1], generated_end)._asdict(),
            }
        )
        output.append(
            {
                "source_segment": bead.segment,
                "source_pose_count": len(poses),
                "stages": [row["stage"] for row in run],
                "joins": joins,
            }
        )
    if snapshot(scene) != before:
        raise WeldSourceJoinError("source analysis changed the scene")
    report = {
        "tour": tour,
        "scene_setup_s_contended": setup,
        "beads": output,
        "scope": "Source bead workpiece poses and already generated lead endpoints. Exact source-line and orientation tangents; machine-scale comparison is not a physical blend tolerance. No original robot joints used as planning seeds.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "tour": tour,
                "beads": len(output),
                "joins": sum(len(item["joins"]) for item in output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
