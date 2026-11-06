"""Select redundant-looking generated samples near protected process transitions."""

import numpy as np

SHORT_INTERVAL_FRACTION = 0.001  # Exploratory fraction of the median modeled sample spacing.
EVENT_TIME_ULPS = 256  # Binary64 arithmetic in modeled seconds, not a physical tolerance.


class GeneratedKnotCoalescingError(RuntimeError):
    """Generated samples, speeds or protected event times are invalid."""


def coalesce_generated_knots(times_s, positions, speed_limits, protected_times_s):
    """Keep transitions and endpoints; remove only a nearby ordinary generated sample.

    A candidate interval must be shorter than the named fraction of the median
    sample spacing, and its generated joint displacement must fit within the
    sourced per-axis speed times its modeled duration. This is an exploratory
    representation change, not a manufacturing tolerance or proof that the
    refitted curve is unchanged.
    """
    times = np.asarray(times_s, dtype=float)
    q = np.asarray(positions, dtype=float)
    speeds = np.asarray(speed_limits, dtype=float)
    protected = np.asarray(protected_times_s, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or q.ndim != 2
        or q.shape[0] != len(times)
        or speeds.shape != (q.shape[1],)
        or not np.isfinite(q).all()
        or not np.isfinite(speeds).all()
        or np.any(speeds <= 0)
        or protected.ndim != 1
        or not np.isfinite(protected).all()
    ):
        raise GeneratedKnotCoalescingError(
            "finite ordered samples, speed scales and events required"
        )
    is_protected = np.zeros(len(times), dtype=bool)
    is_protected[0] = is_protected[-1] = True
    for event in protected:
        band = EVENT_TIME_ULPS * np.finfo(float).eps * max(1.0, abs(float(event)))
        found = np.flatnonzero(np.abs(times - event) <= band)
        if len(found) != 1:
            raise GeneratedKnotCoalescingError(
                "each protected event must match exactly one existing sample"
            )
        is_protected[found[0]] = True
    threshold = SHORT_INTERVAL_FRACTION * float(np.median(np.diff(times)))
    removed = set()
    for index in range(1, len(times) - 1):
        if is_protected[index]:
            continue
        for neighbor in (index - 1, index + 1):
            dt = abs(float(times[index] - times[neighbor]))
            if (
                is_protected[neighbor]
                and dt < threshold
                and np.all(np.abs(q[index] - q[neighbor]) <= speeds * dt)
            ):
                removed.add(index)
                break
    return np.delete(times, sorted(removed)), [float(times[i]) for i in sorted(removed)]
