"""Generate unverified probe curves for an explicit later whole-tour check."""

import numpy as np
from actual_stage_adapters import weld_stage
from actual_stage_contract import Seconds, diagnostic_joins
from master_analytic_ik import arm_ik
from probe_cruise_clock import search_cruise_clock
from probe_cruise_trial import one_sided_progress
from probe_group_coordination import build_probe_group
from probe_shared_velocity import (
    solve_constrained_shared_acceleration,
    solve_constrained_shared_velocity,
)
from rotation_targets import interpolate_rotations
from scipy.interpolate import BPoly
from weld_tangent_leads import fit_lead

SPEED_BOUND_ULPS = 32  # Binary64 allowance at a sourced axis-speed bound.


class ProbeCruiseGroupError(RuntimeError):
    """The generated probe cannot satisfy this explicitly modeled policy."""


def generate_cruise_group(env, opw, info, names, limits7, speeds, phases, timing):
    """Generate a probe; consumers must independently verify its actual curve."""
    speeds = np.asarray(speeds, dtype=float)
    previous, _ = build_probe_group(phases, [float(t["duration_s"]) for t in timing])
    fixture = np.asarray(phases[0]["states"][0][7:], dtype=float)
    qvalues = [np.asarray(row["states"], dtype=float) for row in phases]
    if any(not np.all(q[:, 7:] == fixture) for q in qvalues):
        raise ProbeCruiseGroupError("moving fixture is unsupported")

    def pose(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        return np.linalg.solve(
            np.asarray(state.link_transforms["workpiece_link"].matrix),
            np.asarray(state.link_transforms["nozzle_link"].matrix),
        )

    def jacobian(q):
        state = env.getState(dict(zip(names, map(float, q), strict=True)))
        rotation = np.asarray(state.link_transforms["workpiece_link"].matrix)[:3, :3]
        world = np.asarray(
            env.getJointGroup(info.manipulator).calcJacobian(q[:7], "nozzle_link"), dtype=float
        )
        return np.vstack((rotation.T @ world[:3], rotation.T @ world[3:]))

    vbound = [np.asarray(previous[0].evaluate(Seconds(0)).velocity)[:7]]
    abound = [np.asarray(previous[0].evaluate(Seconds(0)).acceleration)[:7]]
    boundaries = []
    for j in range(3):
        left, right = qvalues[j], qvalues[j + 1]
        if not np.array_equal(left[-1], right[0]):
            raise ProbeCruiseGroupError("generated process endpoints differ")
        q0 = left[-1]
        dtl = float(timing[j]["duration_s"]) / (len(left) - 1)
        dtr = float(timing[j + 1]["duration_s"]) / (len(right) - 1)
        reference = 0.5 * (
            (left[-1, :7] - left[-2, :7]) / dtl + (right[1, :7] - right[0, :7]) / dtr
        )
        J = jacobian(q0)
        scales = np.linalg.norm(J * speeds[None, :], axis=1)
        solution = solve_constrained_shared_velocity(
            J,
            tuple(np.zeros(6)),
            tuple(reference),
            tuple(speeds),
            tuple(scales),
            tuple(q0[:7]),
            tuple(map(tuple, np.asarray(limits7))),
        )
        velocity = np.asarray(solution.velocity)
        magnitude = float(np.max(np.abs(velocity)))
        if magnitude == 0:
            jdot = np.zeros_like(J)
        else:
            dt = np.cbrt(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(q0[:7])))) / magnitude
            qa, qb = q0.copy(), q0.copy()
            qa[:7] -= velocity * dt
            qb[:7] += velocity * dt
            jdot = (jacobian(qb) - jacobian(qa)) / (2 * dt)
        characteristic = min(float(timing[j]["duration_s"]), float(timing[j + 1]["duration_s"]))
        weights = speeds / characteristic  # Relative metric only, not acceleration limits.
        acc_scales = np.linalg.norm(J * weights[None, :], axis=1)
        acceleration_solution = solve_constrained_shared_acceleration(
            J,
            tuple(-(jdot @ velocity)),
            tuple(weights),
            tuple(acc_scales),
            solution.active_lower_axes,
            solution.active_upper_axes,
        )
        acceleration = np.asarray(acceleration_solution.acceleration)
        vbound.append(velocity)
        abound.append(acceleration)
        boundaries.append(
            {
                "left": phases[j]["stage"],
                "right": phases[j + 1]["stage"],
                "velocity": velocity.tolist(),
                "acceleration": acceleration.tolist(),
                "zero_twist_residual": solution.task_residual,
                "active_lower_axes": solution.active_lower_axes,
                "active_upper_axes": solution.active_upper_axes,
                "zero_twist_acceleration_residual": (J @ acceleration + jdot @ velocity).tolist(),
            }
        )
    vbound.append(np.asarray(previous[-1].evaluate(previous[-1].duration).velocity)[:7])
    abound.append(np.asarray(previous[-1].evaluate(previous[-1].duration).acceleration)[:7])
    part = np.asarray(
        env.getState(dict(zip(names[7:], map(float, fixture), strict=True)))
        .link_transforms["workpiece_link"]
        .matrix
    )
    actual, checked = [], []
    for j, (row, source, entry) in enumerate(zip(phases, qvalues, timing, strict=True)):
        source_duration = float(entry["duration_s"])
        if j in (1, 2):
            duration, clock = search_cruise_clock(source_duration)

            def progress(t, clock=clock):
                return clock(t)[0]
        else:
            duration = previous[j].duration

            def progress(t, j=j, duration=duration):
                return one_sided_progress(
                    t, duration, 1.0 if j == 0 else 0.0, 1.0 if j == 3 else 0.0
                )

        times = np.linspace(0, duration, len(source))
        u = np.asarray(progress(times))
        first, last = pose(source[0]), pose(source[-1])
        targets = np.tile(np.eye(4), (len(times), 1, 1))
        targets[:, :3, 3] = first[:3, 3] + u[:, None] * (last[:3, 3] - first[:3, 3])
        targets[:, :3, :3] = interpolate_rotations(first[:3, :3], last[:3, :3], u)
        rail = BPoly.from_derivatives(
            [0.0, float(duration)],
            [
                [source[0, 0], vbound[j][0], abound[j][0]],
                [source[-1, 0], vbound[j + 1][0], abound[j + 1][0]],
            ],
        )
        states = [source[0, :7].copy()]
        branch_counts = []
        for k in range(1, len(times) - 1):
            carrier = dict(
                zip(
                    (names[0], names[7], names[8]),
                    map(float, (rail(times[k]), *fixture)),
                    strict=True,
                )
            )
            solved = arm_ik(
                env,
                opw,
                part @ targets[k],
                carrier,
                tcp_link="nozzle_link",
                limits=np.asarray(limits7)[1:],
            )
            branch_counts.append(len(solved.solutions))
            if not solved.solutions:
                raise ProbeCruiseGroupError(f"{row['stage']} sample {k}: no generated IK branch")
            choices = [np.r_[rail(times[k]), arm] for arm in solved.solutions]
            states.append(
                min(choices, key=lambda q: float(np.sum(((q - states[-1]) / speeds) ** 2)))
            )
        states.append(source[-1, :7].copy())
        points = np.asarray(states)
        if np.any(
            np.abs(np.diff(points, axis=0)) / np.diff(times)[:, None] / speeds
            > 1 + SPEED_BOUND_ULPS * np.finfo(float).eps
        ):
            raise ProbeCruiseGroupError(f"{row['stage']}: generated IK step exceeds sourced speed")
        curve = fit_lead(times, points, (vbound[j], abound[j]), (vbound[j + 1], abound[j + 1]))
        stage = weld_stage(row["stage"], float(duration), curve, 0.0, tuple(fixture))
        checked.append(
            {
                "stage": stage.label,
                "duration_s": float(duration),
                "source_duration_s": source_duration,
                "source_nominal_feed_m_s": float(
                    np.linalg.norm(last[:3, 3] - first[:3, 3]) / source_duration
                ),
                "status": "generated_unverified",
                "min_ik_branches": min(branch_counts),
                "ik_states": points.tolist(),
                "v0": vbound[j].tolist(),
                "v1": vbound[j + 1].tolist(),
                "a0": abound[j].tolist(),
                "a1": abound[j + 1].tolist(),
            }
        )
        actual.append(stage)
    return actual, {
        "process": checked,
        "boundaries": boundaries,
        "process_joins": [item._asdict() for item in diagnostic_joins(actual)],
    }
