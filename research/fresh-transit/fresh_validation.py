"""Independent actual-spline finite-sample checks."""

import numpy as np


class CurveLimitError(RuntimeError):
    """An actual modeled spline exceeds a configured joint position band."""


class CurveCollisionError(RuntimeError):
    """An actual modeled spline sample is in collision."""


class CurveSamplesError(RuntimeError):
    """Samples or the configured bounds cannot be evaluated."""


def inspect_sampled_curve(checker, samples, limits, feasibility_band):
    """Check nine-axis positions; retain raw excursions and reject penetrations.

    The band is an explicit SQP feasibility input in native axis units, not a
    physical tolerance. Collision detection remains finite sampling.
    """
    q = np.asarray(samples, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    if (
        q.ndim != 2
        or q.shape[1] != 9
        or not len(q)
        or not np.isfinite(q).all()
        or bounds.shape != (9, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
        or not np.isfinite(feasibility_band)
        or feasibility_band < 0
    ):
        raise CurveSamplesError("finite nine-axis samples and bounds required")
    excursions = np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0.0)
    maximum = np.max(excursions, axis=0)
    if np.any(maximum > feasibility_band):
        raise CurveLimitError(f"actual spline exceeds configured band: {maximum.tolist()}")
    for index, state in enumerate(q):
        contacts = checker.contacts(state)
        penetration = [float(hit.distance_m) for hit in contacts if float(hit.distance_m) < 0]
        if penetration:
            raise CurveCollisionError(f"actual spline sample {index}: {min(penetration)} m")
    return {
        "sample_count": len(q),
        "feasibility_band_native_units": float(feasibility_band),
        "max_raw_limit_excursion_by_axis": maximum.tolist(),
        "penetrating_samples": 0,
    }
