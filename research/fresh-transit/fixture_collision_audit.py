"""Compare nine-axis native collision values/Jacobians with an independent checker."""

import argparse
import json
from pathlib import Path

import numpy as np
from _spline_dynamic import DynamicSplineCollisionConstraint
from free_space_transit import _ifopt_profile
from fresh_product_scene import initialize_scene, snapshot
from runtime_libraries import require_loaded_libraries
from tesseract_robotics import trajopt_ifopt as ti

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

# Central-difference perturbation in native joint units; a numerical derivative probe.
DIFFERENCE_STEP = 1e-6
# Native default constraint search buffer, metres, also used by the independent checker.
SEARCH_MARGIN_M = 0.01


class FixtureCollisionAuditError(RuntimeError):
    """A comparable moving-fixture contact or collision derivative is unavailable."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "recovery", "output", "ifopt-library", "sqp-library"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise FixtureCollisionAuditError("output already exists")
    args.output.mkdir(parents=True)
    scene = initialize_scene(args.station, args.tour_index, args.output)
    original = snapshot(scene)
    env, _, _, _, names, *_ = scene["inputs"]
    recovery = json.loads(args.recovery.read_text())
    if recovery["verified"]["status"] != "independently_sampled_tour":
        raise FixtureCollisionAuditError("verified generated states required")
    row = next(s for s in recovery["verified"]["expanded"]["segments"] if s["stage"] == args.stage)
    q = np.asarray(row["states"][0], dtype=float)
    group = env.getJointGroup("fixture_audit_nine", list(names))
    active_links = list(group.getActiveLinkNames())
    checker = StateChecker.build(
        env, tuple(names), {}, PairMargins.uniform(Meters(SEARCH_MARGIN_M))
    )
    evaluator = ti.SingleTimestepCollisionEvaluator(
        group, env, _ifopt_profile(True).collision_constraint_config, False
    )
    initial = evaluator.calcCollisions(q, 100)
    if len(initial.gradient_results_sets) != 1:
        raise FixtureCollisionAuditError(
            "expected exactly one native contact for stable row identity"
        )
    contact_set = initial.gradient_results_sets[0]
    pair = tuple(contact_set.key.orderedNameView())
    if "workpiece_link" not in active_links or not all(x in active_links for x in pair):
        raise FixtureCollisionAuditError("both moving robot and fixture links must be active")
    hits = [x for x in checker.contacts(q) if set(x.links) == set(pair)]
    if len(hits) != 1:
        raise FixtureCollisionAuditError("independent checker did not report the same unique pair")
    node = ti.Node("fixture-contact-audit")
    node.addVar("position", list(names), q, [ti.Bounds(-float("inf"), float("inf"))] * 9)
    variables = ti.NodesVariables("fixture-contact-audit", [node])
    var = variables.getNodes()[0].getVar("position")
    term = DynamicSplineCollisionConstraint(evaluator, [var], np.ones(1), np.zeros(9), "audit")
    term.linkWithVariables(variables)
    term.update()
    values = np.asarray(term.getValues())
    jacobian = term.getJacobian().toarray()
    if values.shape != (1,) or jacobian.shape != (1, 9):
        raise FixtureCollisionAuditError("native contact row changed or is not nine-axis")
    comparisons = []
    for axis in (0, 1, 7):
        values_at = []
        distances = []
        for direction in (-1.0, 1.0):
            trial = q.copy()
            trial[axis] += direction * DIFFERENCE_STEP
            variables.setVariables(trial)
            term.update()
            native = np.asarray(term.getValues())
            same = [x for x in checker.contacts(trial) if set(x.links) == set(pair)]
            if native.shape != (1,) or len(same) != 1:
                raise FixtureCollisionAuditError(
                    "contact row or checker pair changed during perturbation"
                )
            values_at.append(float(native[0]))
            distances.append(float(same[0].distance_m))
        comparisons.append(
            {
                "axis": names[axis],
                "native_analytic": float(jacobian[0, axis]),
                "native_finite_difference": (values_at[1] - values_at[0]) / (2 * DIFFERENCE_STEP),
                "independent_distance_finite_difference": (distances[1] - distances[0])
                / (2 * DIFFERENCE_STEP),
            }
        )
    variables.setVariables(q)
    if snapshot(scene) != original:
        raise FixtureCollisionAuditError("scene state changed")
    result = {
        "stage": args.stage,
        "links": pair,
        "both_links_active": True,
        "native_error": float(values[0]),
        "independent_distance_m": float(hits[0].distance_m),
        "comparisons": comparisons,
        "scope": "One real generated state and stable contact row; no continuous collision proof.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
