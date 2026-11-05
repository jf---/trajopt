"""Nine-axis join diagnostics on the same explicit modeled stage clock."""

import numpy as np


class JoinDiagnosticsError(RuntimeError):
    """Stage data, modeled clock, or sampled spline endpoint is invalid."""


def modeled_join_diagnostics(segments, clock, spline_endpoints):
    """Report position and modeled velocity jumps without assigning event time.

    Axis order: track m, six arm rad, positioner rad, tailstock m. Velocities
    are respective units per second. For zero-time events velocity is undefined.
    """
    if len(segments) != len(clock["stages"]):
        raise JoinDiagnosticsError("stage and clock count differ")
    results = []
    for index in range(len(segments) - 1):
        a, b = segments[index : index + 2]
        qa, qb = np.asarray(a["states"], dtype=float), np.asarray(b["states"], dtype=float)
        ta, tb = clock["stages"][index : index + 2]
        da, db = float(ta["duration_s"]), float(tb["duration_s"])
        if ta["stage"] != a["stage"] or tb["stage"] != b["stage"] or da < 0 or db < 0:
            raise JoinDiagnosticsError("stage labels or durations invalid")
        if qa.ndim != 2 or qb.ndim != 2 or qa.shape[1:] != (9,) or qb.shape[1:] != (9,):
            raise JoinDiagnosticsError("nine-axis stage states required")

        def boundary(stage, states, duration, end):
            if stage["stage"] in spline_endpoints:
                return np.asarray(
                    spline_endpoints[stage["stage"]]["end" if end else "start"], dtype=float
                )
            q = states[-1] if end else states[0]
            if duration == 0:
                return q, None
            v = (
                ((states[-1] - states[-2]) if end else (states[1] - states[0]))
                * (len(states) - 1)
                / duration
            )
            return q, v

        left_q, left_v = boundary(a, qa, da, True)
        right_q, right_v = boundary(b, qb, db, False)
        results.append(
            {
                "left": a["stage"],
                "right": b["stage"],
                "position_jump": (right_q - left_q).tolist(),
                "velocity_jump_per_s": None
                if left_v is None or right_v is None
                else (right_v - left_v).tolist(),
                "velocity_status": "undefined_at_zero_time_event"
                if left_v is None or right_v is None
                else "modeled",
                "acceleration_jerk_status": "not_established_for_joint_linear_process",
            }
        )
    return results
