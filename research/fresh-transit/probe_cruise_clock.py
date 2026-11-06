"""Modeled SearchL ramp/cruise/turn clock with source-derived feed."""

from collections.abc import Callable
from math import isfinite
from typing import Any

import numpy as np
from actual_stage_contract import Seconds


class ProbeClockError(RuntimeError):
    """A source duration or modeled clock sample is invalid."""


def search_cruise_clock(
    source_duration_s: float,
) -> tuple[Seconds, Callable[[Any], tuple[Any, Any, Any, Any]]]:
    """Ramp to source feed, cruise, then ramp to zero without a feed overshoot.

    Each smooth ramp lasts half the original modeled duration; the remaining
    half is at the original feed. This global ablation adds half the source
    duration and does not claim observed controller timing or hit behavior.
    Progress is dimensionless and derivatives are per modeled second.
    """
    if not isfinite(source_duration_s) or source_duration_s <= 0:
        raise ProbeClockError("positive source modeled duration required")
    ramp = source_duration_s / 2
    cruise = source_duration_s / 2
    duration = Seconds(2 * ramp + cruise)

    def evaluate(times: Any) -> tuple[Any, Any, Any, Any]:
        t = np.asarray(times, dtype=float)
        if (
            t.ndim != 1
            or not len(t)
            or not np.isfinite(t).all()
            or np.any((t < 0) | (t > duration))
        ):
            raise ProbeClockError("finite clock samples inside modeled duration required")
        progress = np.empty_like(t)
        rate = np.empty_like(t)
        acceleration = np.empty_like(t)
        jerk = np.empty_like(t)
        first = t <= ramp
        middle = (t > ramp) & (t < ramp + cruise)
        last = t >= ramp + cruise

        def ramp_values(u: Any) -> tuple[Any, Any, Any, Any]:
            v = 10 * u**3 - 15 * u**4 + 6 * u**5
            area = 2.5 * u**4 - 3 * u**5 + u**6
            dv = 30 * u**2 - 60 * u**3 + 30 * u**4
            ddv = 60 * u - 180 * u**2 + 120 * u**3
            return v, area, dv, ddv

        v, area, dv, ddv = ramp_values(t[first] / ramp)
        progress[first] = ramp * area / source_duration_s
        rate[first] = v / source_duration_s
        acceleration[first] = dv / (source_duration_s * ramp)
        jerk[first] = ddv / (source_duration_s * ramp**2)
        progress[middle] = (ramp / 2 + t[middle] - ramp) / source_duration_s
        rate[middle] = 1 / source_duration_s
        acceleration[middle] = jerk[middle] = 0.0
        u = (t[last] - ramp - cruise) / ramp
        v, area, dv, ddv = ramp_values(u)
        progress[last] = (ramp / 2 + cruise + ramp * (u - area)) / source_duration_s
        rate[last] = (1 - v) / source_duration_s
        acceleration[last] = -dv / (source_duration_s * ramp)
        jerk[last] = -ddv / (source_duration_s * ramp**2)
        # These are exact boundary conditions of the constructed clock.
        endpoint = t == duration
        progress[endpoint] = 1.0
        rate[endpoint] = acceleration[endpoint] = jerk[endpoint] = 0.0
        return progress, rate, acceleration, jerk

    return duration, evaluate
