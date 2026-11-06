"""Rail reference from generated nine-axis states on their modeled clock."""

from typing import NewType

import numpy as np
import numpy.typing as npt
from actual_stage_contract import Seconds

RailMeters = NewType("RailMeters", float)


class GeneratedRailError(RuntimeError):
    """The generated rail reference is missing or has an invalid clock."""


def generated_rail_at(
    states: npt.NDArray[np.float64], duration_s: Seconds, time_s: Seconds
) -> RailMeters:
    """Interpolate generated rail values, in meters, on a modeled stage clock.

    States have columns track (m), six arm joints (rad), positioner (rad),
    and tailstock (m). This generated reference does not prescribe a physical
    controller trajectory or optimize the rail.
    """
    points = np.asarray(states, dtype=float)
    duration = float(duration_s)
    time = float(time_s)
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] != 9
        or not np.all(np.isfinite(points))
        or not np.isfinite(duration)
        or not np.isfinite(time)
        or duration <= 0.0
        or time < 0.0
        or time > duration
    ):
        raise GeneratedRailError("expected finite nine-axis states and a positive modeled clock")
    node_times = np.linspace(0.0, duration, len(points))
    return RailMeters(float(np.interp(time, node_times, points[:, 0])))
