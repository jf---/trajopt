"""Inspect original Cartesian probe tangents in one prepared scene."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _quintic_spline  # noqa: F401  # Load the native research libraries for path verification.
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401  # Verify the selected SQP runtime.
from fresh_product_scene import initialize_scene, snapshot
from original_probe_search import extract_original_probe_search
from probe_join_geometry import inspect_probe_join
from probe_process_inputs import ProbeRequest, extract_process_tour
from runtime_libraries import require_loaded_libraries
from weld_only_inputs import extract_tour


class ProbeSourceTrialError(RuntimeError):
    """The requested source probe geometry is unavailable or ambiguous."""


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, 3] = pose.translation_m
    result[:3, :3] = pose.rotation
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tour", type=int, action="append", required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise ProbeSourceTrialError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    before = snapshot(scene)
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
    group = env.getJointGroup("source_probe_join_fk", list(names))
    if group is None:
        raise ProbeSourceTrialError("full joint group unavailable")
    phases = ("approach → search", "search → retract", "retract → withdrawal")
    tour_results = []
    for tour in args.tour:
        task = extract_tour(source, tours, tour, "workpiece_link", home=home)
        process = extract_process_tour(program, task, probe_metadata_frame="workpiece_link")
        requests = [request for request in process.operations if isinstance(request, ProbeRequest)]
        if not requests:
            raise ProbeSourceTrialError(f"tour {tour}: original probes unavailable")
        probes = []
        for request in requests:
            original = extract_original_probe_search(program, request, group, "nozzle_link")
            poses = [pose_matrix(pose) for pose in original.poses]
            joins = []
            for index, label in enumerate(phases):
                evidence = inspect_probe_join(poses[index], poses[index + 1], poses[index + 2])
                joins.append({"label": label, **evidence._asdict()})
            probes.append({"source_segment": request.segment, "joins": joins})
        tour_results.append({"tour": tour, "probes": probes})
    if snapshot(scene) != before:
        raise ProbeSourceTrialError("source analysis changed the scene")
    report = {
        "tours": tour_results,
        "scene_setup_s_contended": setup,
        "scope": "Original five workpiece-frame FK poses only; no original intermediate robot joints used to seed planning. Exact pose interpolation tangent evidence with machine-scale band; no physical blend tolerance, measured hit or controller timing.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "tours": args.tour,
                "probes": sum(len(item["probes"]) for item in tour_results),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
