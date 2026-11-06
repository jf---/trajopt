"""Compare actual sampled contacts with the native hard collision rows."""

import numpy as np
from _spline_dynamic import DynamicSplineCollisionConstraint
from free_space_transit import _ifopt_profile
from scipy.interpolate import CubicSpline
from tesseract_robotics import trajopt_ifopt as ti
from weld_actual_witnesses import bracketing_witnesses


class WeldContactAuditError(RuntimeError):
    """A generated bead or native collision row cannot be independently checked."""


def audit_bead_collision(env, info, names, checker, saved, candidate, label, sample_count):
    """Audit worst actual sample, its collocation neighbors and the generated source."""
    matches = [i for i, item in enumerate(candidate["stages"]) if item["stage"] == label]
    if len(matches) != 1 or sample_count < 2:
        raise WeldContactAuditError("one candidate bead and at least two samples required")
    index = matches[0]
    record = candidate["stage_curves"][index]
    knots = np.asarray(record["knots_s"])
    curve = CubicSpline(
        knots, np.asarray(record["nodes"]), axis=0, bc_type=((1, record["v0"]), (1, record["v1"]))
    )
    baseline = CubicSpline(
        np.asarray(saved["times_s"]), np.asarray(saved["joint_states_7"]), axis=0
    )
    fixture = np.asarray(saved["fixture"])
    env.setState(dict(zip(names[7:], map(float, fixture), strict=True)))
    times = np.linspace(knots[0], knots[-1], sample_count)
    contacts = []
    for i, time in enumerate(times):
        q = np.r_[curve(time), fixture]
        for hit in checker.contacts(q):
            if float(hit.distance_m) < 0:
                contacts.append(
                    {
                        "sample": i,
                        "time_s": float(time),
                        "distance_m": float(hit.distance_m),
                        "links": list(hit.links),
                    }
                )
    if not contacts:
        raise WeldContactAuditError("the selected bead has no sampled penetrating contact")
    worst = min(contacts, key=lambda row: row["distance_m"])
    witness_times = candidate["collision_witness_times_s"][index]
    brackets = bracketing_witnesses(witness_times, worst["time_s"])
    group = env.getJointGroup(info.manipulator)
    evaluator = ti.SingleTimestepCollisionEvaluator(
        group, env, _ifopt_profile(True).collision_constraint_config, False
    )
    variables = ti.createNodesVariables(
        "collision-audit",
        list(names[:7]),
        [np.zeros(7)],
        [ti.Bounds(-float("inf"), float("inf"))] * 7,
    )
    var = variables.getNodes()[0].getVar("joints")
    constraint = DynamicSplineCollisionConstraint(
        evaluator, [var], np.ones(1), np.zeros(7), "collision-audit"
    )
    constraint.linkWithVariables(variables)

    def evaluate(time, generated):
        q7 = baseline(time) if generated else curve(time)
        variables.setVariables(q7)
        constraint.update()
        values = np.asarray(constraint.getValues())
        bounds = constraint.getBounds()
        rows = [
            {
                "value": float(value),
                "lower": float(bound.getLower()) if np.isfinite(bound.getLower()) else None,
                "upper": float(bound.getUpper()) if np.isfinite(bound.getUpper()) else None,
            }
            for value, bound in zip(values, bounds, strict=True)
        ]
        hits = checker.contacts(np.r_[q7, fixture])
        return {
            "time_s": float(time),
            "generated_source": generated,
            "rows": rows,
            "contacts": [
                {"links": list(hit.links), "distance_m": float(hit.distance_m)} for hit in hits
            ],
        }

    return {
        "stage": label,
        "sample_count": sample_count,
        "penetrating_samples": len({item["sample"] for item in contacts}),
        "worst_contact": worst,
        "all_contacts": contacts,
        "optimizer_collision_times_s": witness_times,
        "bracketing_optimizer_times_s": list(brackets),
        "candidate_at_worst": evaluate(worst["time_s"], False),
        "candidate_at_brackets": [evaluate(t, False) for t in brackets],
        "generated_at_worst": evaluate(worst["time_s"], True),
    }
