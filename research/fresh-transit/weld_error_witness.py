"""Select bounded actual-curve Cartesian witnesses."""

import numpy as np
import numpy.typing as npt


class WeldWitnessError(RuntimeError):
    """Cartesian witness input is invalid."""


def select_error_witnesses(
    stages: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]],
    budget: int,
) -> list[tuple[int, int]]:
    """Select at most one eligible sample per stage using dimensionless maxima.

    Each input contains position error in metres, orientation error in radians,
    and an eligibility mask. Each error is normalized by the largest eligible
    error of its own kind; this sets a relative priority, not a tolerance.
    """
    if budget < 0:
        raise WeldWitnessError("witness budget cannot be negative")
    maximum_position = 0.0
    maximum_orientation = 0.0
    for position, orientation, eligible in stages:
        if (
            position.ndim != 1
            or orientation.shape != position.shape
            or eligible.shape != position.shape
            or eligible.dtype != np.bool_
            or not np.isfinite(position).all()
            or not np.isfinite(orientation).all()
            or np.any(position < 0.0)
            or np.any(orientation < 0.0)
        ):
            raise WeldWitnessError("matching finite nonnegative Cartesian errors required")
        if np.any(eligible):
            maximum_position = max(maximum_position, float(np.max(position[eligible])))
            maximum_orientation = max(maximum_orientation, float(np.max(orientation[eligible])))
    ranked = []
    for stage, (position, orientation, eligible) in enumerate(stages):
        score = np.zeros_like(position)
        if maximum_position:
            score = np.maximum(score, position / maximum_position)
        if maximum_orientation:
            score = np.maximum(score, orientation / maximum_orientation)
        score[~eligible] = 0.0
        if score.size and np.max(score) > 0.0:
            sample = int(np.argmax(score))
            ranked.append((float(score[sample]), stage, sample))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(stage, sample) for _, stage, sample in ranked[:budget]]
