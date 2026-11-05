"""Fresh Tour 20 planning, native transit refinement, independent assembled check."""

import argparse
import ctypes
import json
from pathlib import Path
from time import perf_counter

import _quintic_spline as native
import numpy as np
import run_multi_episode_shared_scene as planner
import verify_multi_episode_shared_scene as verification
from all24_spline_scene import native_log, prepare_tour
from check_transit_sqp_callback import serializable
from fresh_scene import initialize_scene, snapshot
from native_transits import refine_transits
from verification_route_samples import verification_all_interval_states

root = Path.cwd()
integration = root.parent.parent
original = integration.parent / "r62-smooth-windows"
station = next(
    (original / ".superpowers/r62-master/tmp/.claude/logs/p2/cache/verbotics").glob(
        "*/station.urdf"
    )
).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root / ".superpowers/fresh-native-tour20")
output = parser.parse_args().output
if output.exists():
    raise RuntimeError("output already exists")
output.mkdir(parents=True)
system = ctypes.CDLL(None)
system._dyld_image_count.restype = ctypes.c_uint32
system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
system._dyld_get_image_name.restype = ctypes.c_char_p
images = [
    Path(name.decode()).resolve()
    for i in range(system._dyld_image_count())
    if (name := system._dyld_get_image_name(i))
]
required = {
    "ifopt": (root / "build-ifopt/libtrajopt_ifopt.dylib").resolve(strict=True),
    "sqp": (integration / ".superpowers/piqp-comparison/build-sqp/libtrajopt_sqp.dylib").resolve(
        strict=True
    ),
}
for label, expected in required.items():
    actual = [image for image in images if image.name == expected.name]
    if actual != [expected]:
        raise RuntimeError(f"{label} loaded {actual}, expected {expected}")
setup = perf_counter()
scene = initialize_scene(station, original, output)
setup_s = perf_counter() - setup
snapshot0 = snapshot(scene)
env, program, source, tours, names, opw, limits, speeds, rails, home, info, velocities = scene[
    "inputs"
]
total = perf_counter()
start = perf_counter()
record = planner.run(
    *scene["inputs"],
    tour=20,
    use_cached_graph=True,
    reuse_probe_rungs=True,
    graph_budget_s=120,
    captured_frontier=None,
    route_transits=True,
    transit_backend="ompl_piqp",
    route_fixtures=True,
    fixture_backend="home_park_search",
    reuse_home_sweeps=True,
    fixture_home_only=True,
    feasibility_first=True,
    defer_unresolved_edges=True,
    validate_weld_intervals=True,
)
planning_s = perf_counter() - start
if record["status"] != "sampled_nodes_only":
    raise RuntimeError(f"fresh planning failed: {record['status']}")
start = perf_counter()
checked = verification.verify(*scene["verify_inputs"], record, sample_all_intervals=True)
initial_check_s = perf_counter() - start
candidate = output / "candidate.json"
independent = output / "initial-check.json"
candidate.write_text(json.dumps(serializable(record), allow_nan=False))
independent.write_text(json.dumps(serializable(checked), allow_nan=False))
raw = json.loads((station / "programs/weld_trajectory.json").read_text())
metadata = json.loads((station / "joint_speeds.json").read_text())["joint_speeds"]["joints"]
start = perf_counter()
segments, clock, sourced_speeds = prepare_tour(
    env, program, source, tours, home, names, raw, metadata, 20, candidate, independent
)
with native_log(output / "native-sqp.log"):
    outcomes = refine_transits(env, info, names, segments, clock, sourced_speeds)
refinement_s = perf_counter() - start
for row in outcomes:
    index = int(row["stage"].split(":")[1])
    edge = record["result"]["edges"][index]
    fixture = np.array(row["fixture"])
    model = native.QuinticSplineModel.build(
        row["knots_s"], np.asarray(row["v0"]), np.zeros(7), np.asarray(row["v1"]), np.zeros(7)
    )
    sample = model.evaluate(np.linspace(0, row["duration_s"], 1025), np.asarray(row["nodes"]), 1)
    full = np.column_stack((sample[0], np.tile(fixture, (1025, 1))))
    row["endpoint_residual"] = max(
        float(np.max(np.abs(full[k] - edge["states"][k]))) for k in (0, -1)
    )
    row["sampled_peak_speed_ratio"] = float(
        np.max(np.max(np.abs(sample[1]), axis=0) / sourced_speeds)
    )
    full[0] = edge["states"][0]
    full[-1] = edge["states"][-1]
    edge["states"] = full.tolist()
    edge.pop("route_cost", None)


def samples(kind, edge, count):
    if kind == "transit":
        yield from np.asarray(edge["states"], dtype=float)
    else:
        yield from verification_all_interval_states(kind, edge, count)


verification.verification_all_interval_states = samples
verification.OUTPUT = output / "assembled.json"
start = perf_counter()
assembled = verification.verify(*scene["verify_inputs"], record, sample_all_intervals=True)
assembled_check_s = perf_counter() - start
if snapshot(scene) != snapshot0:
    raise RuntimeError("scene state was not restored")
result = {
    "scope": "Fresh no-recorded-intermediate-seed plan; opt-in native transit splines and independent sampled check. Fixed modeled clock; no full-tour C1 or physical timing claim.",
    "native_libraries": {key: str(value) for key, value in required.items()},
    "scene_setup_s": setup_s,
    "planning_s": planning_s,
    "initial_check_s": initial_check_s,
    "refinement_s": refinement_s,
    "assembled_check_s": assembled_check_s,
    "elapsed_after_scene_s": perf_counter() - total,
    "outcomes": outcomes,
    "assembled_samples": assembled["samples_checked"],
    "probe_checks": assembled["probe_checks"],
    "weld_max_position_error_m": assembled["weld_max_position_error_m"],
    "weld_max_orientation_error_rad": assembled["weld_max_orientation_error_rad"],
}
(output / "summary.json").write_text(
    json.dumps(serializable(result), indent=2, allow_nan=False) + "\n"
)
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("outcomes", "probe_checks")}
        | {
            "transits": len(outcomes),
            "statuses": [r["status"] for r in outcomes],
            "maximum_sampled_speed_ratio": max(r["sampled_peak_speed_ratio"] for r in outcomes),
        }
    ),
    flush=True,
)
