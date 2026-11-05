"""Fixed-duration monotone quintic progress on generated joint geometry."""

import numpy as np
from numpy.polynomial import Polynomial

# Floating point root and bound checks, not physical process tolerances.
ROUNDING_ULPS = 128


class ProgressClockError(RuntimeError):
    """The modeled process clock has invalid input or reverses progress."""


def quintic_progress(times_s, duration_s, start_rate, end_rate):
    """Return dimensionless progress and three time derivatives.

    Endpoint rates are dimensionless derivatives with respect to normalized
    time. Endpoint progress accelerations are zero. The same duration is kept;
    the minimum rate is checked at every real stationary point, not just samples.
    """
    t = np.asarray(times_s, dtype=float)
    duration = float(duration_s)
    rates = np.asarray([start_rate, end_rate], dtype=float)
    if (
        t.ndim != 1
        or len(t) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(duration)
        or duration <= 0
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
        or np.any(np.diff(t) < 0)
        or np.any(t < 0)
        or np.any(t > duration)
    ):
        raise ProgressClockError(
            "finite ordered in-domain times, positive duration and rates required"
        )
    v0, v1 = rates
    polynomial = Polynomial(
        [0, v0, 0, 10 - 6 * v0 - 4 * v1, -15 + 8 * v0 + 7 * v1, 6 - 3 * v0 - 3 * v1]
    )
    rate = polynomial.deriv()
    epsilon = ROUNDING_ULPS * np.finfo(float).eps * max(1.0, float(np.max(np.abs(rate.coef))))
    extrema = rate.deriv().roots()
    real = extrema[np.abs(extrema.imag) <= epsilon].real
    points = np.r_[0.0, 1.0, real[(real > 0) & (real < 1)]]
    if np.min(rate(points)) < -epsilon:
        raise ProgressClockError("modeled process progress reverses between endpoints")
    tau = t / duration
    return tuple(polynomial.deriv(i)(tau) / duration**i for i in range(4))


def compose_process(curve, times_s, duration_s, start_rate, end_rate):
    """Evaluate generated joint geometry and q, qd, qdd, qddd on this clock.

    Joint positions follow the curve's native axis units; derivatives divide by
    seconds to the corresponding order. This is an interpolation model.
    """
    s, sd, sdd, sddd = quintic_progress(times_s, duration_s, start_rate, end_rate)
    q = curve(s)
    qs, qss, qsss = (curve(s, i) for i in (1, 2, 3))
    v = qs * sd[:, None]
    a = qss * sd[:, None] ** 2 + qs * sdd[:, None]
    j = qsss * sd[:, None] ** 3 + 3 * qss * sd[:, None] * sdd[:, None] + qs * sddd[:, None]
    return q, v, a, j
