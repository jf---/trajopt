"""Compare all generated IK branches at saved tangent-lead target nodes."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _quintic_spline  # noqa: F401  # Load selected native research libraries.
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401  # Verify selected SQP runtime.
from fresh_product_scene import initialize_scene, snapshot
from master_analytic_ik import arm_ik
from runtime_libraries import require_loaded_libraries

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

NODE_TIME_ULPS = 16  # Same modeled-time equality band as the generated node set.


class WeldLeadBranchAuditError(RuntimeError):
    """Saved generated targets or their scene are unavailable."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "input",
        "weld-evidence",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    candidate = json.loads(args.input.read_text())["result"]["jerk"]
    evidence = json.loads(args.weld_evidence.read_text())
    if args.output.exists():
        raise WeldLeadBranchAuditError("output already exists")
    args.output.mkdir(parents=True)
    began = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup = perf_counter() - began
    before = snapshot(scene)
    env, _, _, _, names, opw, limits7, _, _, _, _, _ = scene["inputs"]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    clock = {row["stage"]: float(row["duration_s"]) for row in candidate["clock"]["stages"]}
    report = {
        "scope": "All generated IK branches at each saved tangent-lead node, fixed generated rail and fixture; state collision checks only. Connections and continuous clearance are not established.",
        "scene_setup_s_contended": setup,
        "beads": [],
    }
    for bead in evidence["beads"]:
        if "joint_states_7" not in bead:
            raise WeldLeadBranchAuditError(
                f"weld {bead['source_segment']}: missing saved generated IK nodes"
            )
        fixture = np.asarray(bead["fixture"], dtype=float)
        times = np.asarray(bead["times_s"], dtype=float)
        states = np.asarray(bead["joint_states_7"], dtype=float)
        stages = bead["stages"]
        durations = [clock[row["stage"]] for row in stages]
        edges = np.r_[0.0, np.cumsum(durations)]
        part = np.asarray(
            env.getState(dict(zip(names[7:], map(float, fixture), strict=True)))
            .link_transforms["workpiece_link"]
            .matrix
        )
        result = {"source_segment": bead["source_segment"], "leads": []}
        for index in (0, len(stages) - 1):
            band = (
                NODE_TIME_ULPS
                * np.finfo(float).eps
                * max(1.0, abs(float(edges[index])), abs(float(edges[index + 1])))
            )
            selected = np.flatnonzero(
                (times >= edges[index] - band) & (times <= edges[index + 1] + band)
            )
            nodes = []
            for node in selected:
                q = states[node]
                full = np.r_[q, fixture]
                state = env.getState(dict(zip(names, map(float, full), strict=True)))
                target = np.linalg.solve(
                    np.asarray(state.link_transforms["workpiece_link"].matrix),
                    np.asarray(state.link_transforms["nozzle_link"].matrix),
                )
                carrier = dict(
                    zip((names[0], names[7], names[8]), map(float, (q[0], *fixture)), strict=True)
                )
                answer = arm_ik(
                    env,
                    opw,
                    part @ target,
                    carrier,
                    tcp_link="nozzle_link",
                    limits=np.asarray(limits7)[1:],
                )
                branches = []
                for arm in answer.solutions:
                    candidate_state = np.r_[q[0], arm, fixture]
                    hits = [
                        {
                            "distance_m": float(hit.distance_m),
                            "links": [str(link) for link in hit.links],
                        }
                        for hit in checker.contacts(candidate_state)
                        if float(hit.distance_m) < 0
                    ]
                    branches.append(
                        {
                            "joint_state_7": candidate_state[:7].tolist(),
                            "clear_at_node": not hits,
                            "worst_contact": min(hits, key=lambda hit: hit["distance_m"])
                            if hits
                            else None,
                        }
                    )
                original_hits = [
                    {
                        "distance_m": float(hit.distance_m),
                        "links": [str(link) for link in hit.links],
                    }
                    for hit in checker.contacts(full)
                    if float(hit.distance_m) < 0
                ]
                nodes.append(
                    {
                        "modeled_time_s": float(times[node]),
                        "branch_count": len(branches),
                        "clear_branch_count": sum(branch["clear_at_node"] for branch in branches),
                        "chosen_clear": not original_hits,
                        "chosen_worst_contact": min(
                            original_hits, key=lambda hit: hit["distance_m"]
                        )
                        if original_hits
                        else None,
                        "branches": branches,
                    }
                )
            if not nodes:
                raise WeldLeadBranchAuditError(f"{stages[index]['stage']}: no saved target nodes")
            result["leads"].append(
                {
                    "stage": stages[index]["stage"],
                    "nodes": nodes,
                    "all_nodes_have_a_clear_branch": all(
                        node["clear_branch_count"] > 0 for node in nodes
                    ),
                    "connections_checked": False,
                }
            )
        report["beads"].append(result)
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "source_segment": bead["source_segment"],
                    "leads": [
                        {
                            "stage": lead["stage"],
                            "nodes": len(lead["nodes"]),
                            "min_clear_branches": min(
                                node["clear_branch_count"] for node in lead["nodes"]
                            ),
                        }
                        for lead in result["leads"]
                    ],
                }
            ),
            flush=True,
        )
    if snapshot(scene) != before:
        raise WeldLeadBranchAuditError("scene state changed")


if __name__ == "__main__":
    main()
