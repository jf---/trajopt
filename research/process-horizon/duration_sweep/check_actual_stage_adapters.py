"""Check adapters against the independently sampled modeled Tour09 assembly."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from actual_stage_contract import diagnostic_joins
from fresh_prepare import prepare_fresh
from fresh_scene import initialize_scene
from fresh_tour_pipeline import SPEED_BOUND_ULPS, SQP_FEASIBILITY_BAND
from runtime_libraries import require_loaded_libraries
from verify_duration_policy import verify_duration_policy

# Floating-point agreement only, in native modeled axis units and derivatives.
ADAPTER_PARITY_ULPS = 32768


class AdapterVerificationError(RuntimeError):
    """Actual stage adapters disagree with independently evaluated endpoints."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "original-worktree",
        "fresh-route",
        "verified-route",
        "artifacts",
        "fixture-policy",
        "output",
        "ifopt-library",
        "sqp-library",
        "transit-summary",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise AdapterVerificationError("output directory already exists")
    args.output.mkdir(parents=True)
    raw = json.loads((args.station / "programs/weld_trajectory.json").read_text())
    metadata = json.loads((args.station / "joint_speeds.json").read_text())["joint_speeds"][
        "joints"
    ]
    started = perf_counter()
    scene = initialize_scene(args.station, args.original_worktree, args.output)
    env, program, source, tours, names, _, _, _, _, home, _, _ = scene["inputs"]
    stages, clock, speed = prepare_fresh(
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
    policy_dir = args.output / "global-duration-2"
    policy_dir.mkdir()
    (policy_dir / "summary.json").write_text(args.transit_summary.read_text())
    adapters = []
    checked = verify_duration_policy(
        2.0,
        scene,
        stages,
        clock,
        speed,
        args.output,
        args.artifacts,
        args.fixture_policy,
        stage_sink=adapters.append,
    )
    if checked["negative_contacts"] or max(checked["raw_limit_excess"]) > SQP_FEASIBILITY_BAND:
        raise AdapterVerificationError(
            "sampled collision or configured position-limit check failed"
        )
    speed_bound = 1.0 + SPEED_BOUND_ULPS * np.finfo(float).eps
    if any(row["max_sourced_arm_track_speed_ratio"] > speed_bound for row in checked["stages"]):
        raise AdapterVerificationError("sampled sourced speed check failed")
    if len(adapters) != len(checked["stages"]):
        raise AdapterVerificationError("adapter stage coverage differs from sampled verifier")
    parity = {"position": 0.0, "velocity": 0.0, "acceleration": 0.0}
    for stage, row in zip(adapters, checked["stages"], strict=True):
        if stage.label != row["stage"]:
            raise AdapterVerificationError("adapter and verifier stage order differs")
        for time, prefix in ((0.0, "first"), (stage.duration, "last")):
            state = stage.evaluate(time)
            for name, actual, field in (
                ("position", state.position, "q"),
                ("velocity", state.velocity, "qd"),
                ("acceleration", state.acceleration, "qdd"),
            ):
                if actual is None:
                    raise AdapterVerificationError(f"{stage.label}: {name} is unknown")
                expected = np.asarray(row[f"{prefix}_{field}"], dtype=float)
                value = np.asarray(actual, dtype=float)
                error = float(np.max(np.abs(expected - value)))
                parity[name] = max(parity[name], error)
                band = (
                    ADAPTER_PARITY_ULPS
                    * np.finfo(float).eps
                    * max(1.0, float(np.max(np.abs(expected))))
                )
                if error > band:
                    raise AdapterVerificationError(f"{stage.label}: {name} adapter mismatch")
    joins = diagnostic_joins(adapters)
    output = {
        "scope": "Generated Tour09 modeled curves; sampled collision and process checks, actual-curve adapter endpoint parity and nine-axis join diagnostics. Outer derivatives unspecified; no controller or manufacturing acceptance.",
        "stage_count": len(adapters),
        "sampled_state_count": checked["checked_samples"],
        "negative_contacts": checked["negative_contacts"],
        "raw_limit_excess": checked["raw_limit_excess"],
        "modeled_duration_s": checked["modeled_tour_duration_s"],
        "endpoint_parity_max_by_derivative": parity,
        "join_count": len(joins),
        "max_join_position_gap": max(max(map(abs, j.position_delta)) for j in joins),
        "max_join_velocity_gap": max(
            max(map(abs, j.velocity_delta)) for j in joins if j.velocity_delta is not None
        ),
        "max_join_acceleration_gap": max(
            max(map(abs, j.acceleration_delta)) for j in joins if j.acceleration_delta is not None
        ),
        "unknown_internal_derivatives": sum(
            j.velocity_delta is None or j.acceleration_delta is None for j in joins
        ),
        "outer_derivatives": "unknown; no controller or endpoint velocity supplied",
        "elapsed_s_including_scene_setup": perf_counter() - started,
        "overall_accepted": False,
    }
    (args.output / "actual-stage-parity.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
