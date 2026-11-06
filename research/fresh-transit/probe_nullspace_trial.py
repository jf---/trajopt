"""Inspect generated zero-TCP nullspace motion at one SearchL reversal."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _quintic_spline  # noqa: F401  # Load the native research libraries for path verification.
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401  # Verify the selected SQP runtime.
from fresh_product_scene import initialize_scene, snapshot
from fresh_tour_pipeline import SQP_FEASIBILITY_BAND
from fresh_validation import inspect_sampled_curve
from probe_shared_velocity import solve_shared_velocity
from runtime_libraries import require_loaded_libraries

from tesseract_qt_py.core.math_utils import rotation_delta_rotvec, rotation_geodesic_angle
from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Central-difference error is O(eps^(2/3)); this multiplier accounts for the FK chain.
JACOBIAN_ROUNDOFF_MULTIPLIER = 64
LOCAL_SAMPLES = 1025  # Same independent density as the other actual-curve checks.


class ProbeNullspaceTrialError(RuntimeError):
    """A generated reversal or its modeled Jacobian cannot be checked."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "input", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--probe", type=int, required=True)
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
        raise ProbeNullspaceTrialError("independently verified fresh candidate required")
    if args.output.exists():
        raise ProbeNullspaceTrialError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    before = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, info, _ = scene["inputs"]
    rows, clock = candidate["generated_stages"], candidate["clock"]["stages"]
    i = next(i for i, row in enumerate(rows) if row["stage"] == f"search:{args.probe}")
    if rows[i + 1]["stage"] != f"retract:{args.probe}":
        raise ProbeNullspaceTrialError("ordered SearchL and retract required")
    search, retract = (
        np.asarray(rows[i]["states"], dtype=float),
        np.asarray(rows[i + 1]["states"], dtype=float),
    )
    if not np.array_equal(search[-1], retract[0]):
        raise ProbeNullspaceTrialError("generated reversal state mismatch")
    q0 = search[-1]
    fixture = q0[7:]
    if not np.all(search[:, 7:] == fixture) or not np.all(retract[:, 7:] == fixture):
        raise ProbeNullspaceTrialError("moving fixture is unsupported")
    dt_in = float(clock[i]["duration_s"]) / (len(search) - 1)
    dt_out = float(clock[i + 1]["duration_s"]) / (len(retract) - 1)
    reference = 0.5 * (
        (search[-1, :7] - search[-2, :7]) / dt_in + (retract[1, :7] - retract[0, :7]) / dt_out
    )
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    speeds = []
    for name in names[:7]:
        entry = metadata[name]
        if entry["speed_source"] not in ("datasheet", "cell_declared"):
            raise ProbeNullspaceTrialError(f"missing sourced speed for {name}")
        speeds.append(float(entry["velocity"]))
    speeds = np.asarray(speeds)

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    state = env.getState(dict(zip(names, map(float, q0), strict=True)))
    workpiece_rotation = np.asarray(state.link_transforms["workpiece_link"].matrix)[:3, :3]
    group = env.getJointGroup(info.manipulator)
    jacobian_world = np.asarray(group.calcJacobian(q0[:7], "nozzle_link"), dtype=float)
    jacobian = np.vstack(
        (workpiece_rotation.T @ jacobian_world[:3], workpiece_rotation.T @ jacobian_world[3:])
    )
    eps = np.finfo(float).eps
    steps = np.cbrt(eps) * np.maximum(1.0, np.abs(q0[:7]))
    difference = np.zeros((6, 7))
    for k, step in enumerate(steps):
        left, right = q0.copy(), q0.copy()
        left[k] -= step
        right[k] += step
        a, b = pose(left), pose(right)
        difference[:3, k] = (b[:3, 3] - a[:3, 3]) / (2 * step)
        difference[3:, k] = rotation_delta_rotvec(a[:3, :3], b[:3, :3]) / (2 * step)
    twist_scales = np.linalg.norm(jacobian * speeds[None, :], axis=1)
    if np.any(twist_scales <= 0):
        raise ProbeNullspaceTrialError("Jacobian has an unsupported zero task row")
    normalized = jacobian * speeds[None, :] / twist_scales[:, None]
    normalized_difference = difference * speeds[None, :] / twist_scales[:, None]
    jacobian_error = float(np.max(np.abs(normalized - normalized_difference)))
    jacobian_band = (
        JACOBIAN_ROUNDOFF_MULTIPLIER
        * eps ** (2 / 3)
        * max(1.0, float(np.max(np.abs(normalized))), float(np.max(np.abs(normalized_difference))))
    )
    if jacobian_error > jacobian_band:
        raise ProbeNullspaceTrialError(
            f"scaled workpiece Jacobian differs from FK finite differences: {jacobian_error}, band {jacobian_band}"
        )
    solution = solve_shared_velocity(
        jacobian, tuple(np.zeros(6)), tuple(reference), tuple(speeds), tuple(twist_scales)
    )
    velocity = np.asarray(solution.velocity)
    span = 0.5 * min(dt_in, dt_out)  # Half of the source's generated sample interval.
    times = np.linspace(-span, span, LOCAL_SAMPLES)
    q = np.tile(q0, (len(times), 1))
    q[:, :7] += times[:, None] * velocity
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    limits = np.asarray(
        env.getJointGroup("probe_nullspace_nine", list(names)).getLimits().joint_limits
    )
    checked = inspect_sampled_curve(checker, q, limits, SQP_FEASIBILITY_BAND)
    poses = [pose(point) for point in q]
    target = pose(q0)
    displacement = np.asarray([np.linalg.norm(p[:3, 3] - target[:3, 3]) for p in poses])
    orientation = np.asarray([rotation_geodesic_angle(target[:3, :3], p[:3, :3]) for p in poses])
    report = {
        "scope": "One generated SearchL reversal; generated reference, fixed fixture, zero TCP twist, straight joint-space local model only. Not a full process path or controller command.",
        "status": solution.status,
        "scene_setup_s_contended": setup,
        "probe": args.probe,
        "generated_reference_velocity_7": reference.tolist(),
        "modeled_velocity_7": velocity.tolist(),
        "jacobian_normalized_max_error": jacobian_error,
        "jacobian_machine_scaled_band": jacobian_band,
        "jacobian_linear_raw_max_error": float(np.max(np.abs(jacobian[:3] - difference[:3]))),
        "jacobian_angular_raw_max_error": float(np.max(np.abs(jacobian[3:] - difference[3:]))),
        "task_residual": solution.task_residual,
        "sourced_speed_ratio": solution.sourced_speed_ratio,
        "modeled_local_half_span_s": span,
        "sampled_collision_and_position": checked,
        "sampled_max_tcp_drift_m": float(displacement.max()),
        "sampled_max_orientation_drift_rad": float(orientation.max()),
        "full_neighborhood_continuity": "not_evaluated",
        "intended_search_feed": "not_evaluated",
        "actual_hit_and_physical_boundary": "unknown",
    }
    if snapshot(scene) != before:
        raise ProbeNullspaceTrialError("scene state changed")
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "local_half_span_s": span,
                "tcp_drift_m": report["sampled_max_tcp_drift_m"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
