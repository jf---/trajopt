"""Replay a generated probe and adjacent transits in one retained scene.

This is an explicitly opt-in research driver. Its input route and independent
verification must come from the fresh planner, not recorded intermediate joints.
"""

import argparse
import code
import ctypes
import json
from pathlib import Path
from time import perf_counter

from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene, snapshot
from ik_progress import run_ik_progress
from ik_progress_retime import run_ik_progress_retime
from probe_neighbor_refinement import run_probe_neighbors

parser = argparse.ArgumentParser()
parser.add_argument("--station", type=Path, required=True)
parser.add_argument("--original-worktree", type=Path, required=True)
parser.add_argument("--fresh-route", type=Path, required=True)
parser.add_argument("--verified-route", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ifopt-library", type=Path, required=True)
parser.add_argument("--sqp-library", type=Path, required=True)
parser.add_argument("--retain-scene", action="store_true")
args = parser.parse_args()
experiment_dir = Path(__file__).resolve().parent
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
dyld._dyld_get_image_name.restype = ctypes.c_char_p
loaded = [
    Path(value.decode()).resolve()
    for index in range(dyld._dyld_image_count())
    if (value := dyld._dyld_get_image_name(index))
]
for library in (args.ifopt_library, args.sqp_library):
    expected = library.resolve(strict=True)
    actual = [path for path in loaded if path.name == expected.name]
    if actual != [expected]:
        raise RuntimeError(f"wrong loaded solver library: expected {expected}, got {actual}")
station = args.station.resolve(strict=True)
original = args.original_worktree.resolve(strict=True)
raw = json.loads((station / "programs/weld_trajectory.json").read_text())
metadata = json.loads((station / "joint_speeds.json").read_text())["joint_speeds"]["joints"]
started = perf_counter()
scene = initialize_scene(station, original, output)
setup_s = perf_counter() - started
before = snapshot(scene)
env, program, source, tours, names, _, _, _, _, home, info, _ = scene["inputs"]
record = json.loads(args.fresh_route.read_text())
checked = json.loads(args.verified_route.read_text())
segments, clock, speed = prepare_fresh(
    env, program, source, tours, home, names, raw, metadata, 9, record, checked
)
print(json.dumps({"scene_setup_s": setup_s, "fresh_tour": 9}), flush=True)
run_ik_progress(scene, segments, clock, speed, output)
run_ik_progress_retime(scene, segments, clock, speed, output)
run_probe_neighbors(scene, segments, clock, speed, output)
if args.retain_scene:
    code.interact(local=globals())
