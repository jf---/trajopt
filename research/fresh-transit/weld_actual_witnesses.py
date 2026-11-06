"""Select bounded witnesses from independently sampled actual process curves."""

import numpy as np


class WeldWitnessError(RuntimeError):
    """The sampled clocks, process errors, or contact distances are invalid."""


def bracketing_witnesses(times_s, time_s):
    """Return the neighboring checked times around an actual-curve sample."""
    times = np.asarray(times_s, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.isfinite(time_s)
        or np.any(np.diff(times) <= 0)
        or time_s < times[0]
        or time_s > times[-1]
    ):
        raise WeldWitnessError("finite ordered witnesses must contain the sample time")
    index = int(np.searchsorted(times, time_s))
    if times[index] == time_s:
        return float(time_s), float(time_s)
    return float(times[index - 1]), float(times[index])


def select_weld_witnesses(times_s, position_errors_m, signed_distances_m, chord_progress_m):
    """Select worst actual process and contact samples, including a reversed step's ends."""
    if (
        not times_s
        or len(times_s) != len(position_errors_m)
        or len(times_s) != len(signed_distances_m)
        or len(times_s) != len(chord_progress_m)
    ):
        raise WeldWitnessError("sampled stage counts differ")
    best_position = (-1.0, None, None)
    worst_contact = (0.0, None, None)
    worst_reverse = (0.0, None, None, None)
    for index, (times, errors, contacts, progress) in enumerate(
        zip(times_s, position_errors_m, signed_distances_m, chord_progress_m, strict=True)
    ):
        t, e, c = np.asarray(times), np.asarray(errors), np.asarray(contacts)
        p = np.asarray(progress)
        if (
            t.ndim != 1
            or not t.size
            or t.shape != e.shape
            or t.shape != c.shape
            or t.shape != p.shape
            or not np.isfinite(t).all()
            or not np.isfinite(e).all()
            or np.isnan(c).any()
            or not np.isfinite(p).all()
            or np.any(e < 0)
        ):
            raise WeldWitnessError("finite matching sample clocks and errors required")
        worst_error_index = int(np.argmax(e))
        worst_contact_index = int(np.argmin(c))
        if e[worst_error_index] > best_position[0]:
            best_position = (float(e[worst_error_index]), index, float(t[worst_error_index]))
        if c[worst_contact_index] < worst_contact[0]:
            worst_contact = (float(c[worst_contact_index]), index, float(t[worst_contact_index]))
        if len(p) > 1:
            diff = np.diff(p)
            k = int(np.argmin(diff))
            if diff[k] < worst_reverse[0]:
                worst_reverse = (float(diff[k]), index, float(t[k]), float(t[k + 1]))
    selected = {}
    for _, index, time in (best_position, worst_contact):
        if index is not None:
            selected.setdefault(index, set()).add(time)
    _, index, start, end = worst_reverse
    if index is not None:
        selected.setdefault(index, set()).update((start, end))
    return {index: sorted(times) for index, times in selected.items()}
