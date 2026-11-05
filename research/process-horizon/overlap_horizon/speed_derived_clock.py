"""Bounded duration refinement based on actual spline velocity extrema."""

import math
import sys

# Floating-point band on a dimensionless ratio; no physical slack is added.
SPEED_ROUNDOFF_ULPS = 256
DEFAULT_MAX_ATTEMPTS = 12


class SpeedClockError(RuntimeError):
    """A sourced-speed clock cannot be obtained within the stated policy."""


def refine_speed_clock(initial_duration_s, solve, *, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Reoptimize after scaling time by the measured peak/sourced speed ratio.

    The solver must return its modeled duration and numerical continuous-spline
    extrema ratio. The same multiplicative rule and budget apply to each transit.
    Success establishes neither minimal duration nor physical controller timing.
    """
    duration = float(initial_duration_s)
    if (
        not math.isfinite(duration)
        or duration <= 0
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise SpeedClockError("positive finite duration and attempt budget required")
    attempts = []
    for _ in range(max_attempts):
        result = solve(duration)
        actual = float(result["duration_s"])
        ratio = float(result["peak_ratio"])
        if actual != duration or not math.isfinite(ratio) or ratio < 0:
            raise SpeedClockError("solver must return matching duration and finite speed ratio")
        attempts.append(result)
        if ratio <= 1.0 + SPEED_ROUNDOFF_ULPS * sys.float_info.epsilon:
            return result, attempts
        duration *= ratio
        if not math.isfinite(duration):
            raise SpeedClockError("speed-based duration overflow")
    raise SpeedClockError(
        f"speed-derived clock did not converge after {max_attempts} attempts: {[(x['duration_s'], x['peak_ratio']) for x in attempts]}"
    )
