"""Select observed out-of-range and backward process-progress witnesses."""

import numpy as np

# Binary64 equality allowance for modeled times and progress increments.
ROUNDOFF_ULPS = 16


class ProgressWitnessError(RuntimeError):
    """Sampled clock, progress or witness configuration is invalid."""


class ProgressWitnessStalled(RuntimeError):
    """Material sampled violations remain only at existing witnesses."""


def select_progress_witnesses(times, progress, existing, progress_band):
    """Select each material backward episode's ends and range-error extrema.

    Progress is dimensionless and clock values are modeled seconds. The
    caller provides the progress band; it is not a manufacturing tolerance.
    """
    t = np.asarray(times, dtype=float)
    p = np.asarray(progress, dtype=float)
    prior = np.asarray(existing, dtype=float)
    if (
        t.ndim != 1
        or len(t) < 2
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or p.shape != t.shape
        or not np.isfinite(p).all()
        or prior.ndim != 1
        or not np.isfinite(prior).all()
        or np.any(prior < t[0])
        or np.any(prior > t[-1])
        or not np.isfinite(progress_band)
        or progress_band < 0
    ):
        raise ProgressWitnessError("finite ordered clock, progress and in-leg witnesses required")
    indices = []
    # A complete decreasing episode matters, not one roundoff-sized difference.
    slope_band = (
        ROUNDOFF_ULPS
        * np.finfo(float).eps
        * np.maximum(1.0, np.maximum(np.abs(p[:-1]), np.abs(p[1:])))
    )
    backward = np.diff(p) < -slope_band
    first = None
    for i, active in enumerate(backward):
        if active and first is None:
            first = i
        if first is not None and (not active or i == len(backward) - 1):
            end = i + 1 if active else i
            if p[first] - p[end] > progress_band:
                indices.extend((first, end))
            first = None
    for error in (np.maximum(-p - progress_band, 0), np.maximum(p - 1 - progress_band, 0)):
        first = None
        for i, value in enumerate(error):
            if value > 0 and first is None:
                first = i
            if first is not None and (value <= 0 or i == len(error) - 1):
                stop = i + 1 if value > 0 else i
                indices.append(first + int(np.argmax(error[first:stop])))
                first = None
    selected = []
    for i in sorted(set(indices)):
        value = float(t[i])
        band = ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, abs(value))
        if not np.any(np.abs(prior - value) <= band):
            selected.append(value)
    if indices and not selected:
        raise ProgressWitnessStalled("material sampled progress violations already constrained")
    return tuple(selected)
