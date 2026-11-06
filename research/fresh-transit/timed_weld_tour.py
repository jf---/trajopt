"""Convert a fresh generated bead to the stage-local optimizer input."""

from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from typing import Any

import numpy as np
from free_space_transit import _restore_scene


class TimedWeldTourError(RuntimeError):
    """A generated bead cannot preserve its modeled clock and fixture."""


def generated_bead_input(row: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Keep generated seven-axis states, fixed fixture and modeled seconds.

    Axis order is track metres, six arm radians, positioner radians, tailstock
    metres. No recorded robot states are introduced by this conversion.
    """
    states = np.asarray(row["states"], dtype=float)
    label = row["stage"]
    if (
        not isinstance(label, str)
        or not label.startswith("weld:")
        or states.ndim != 2
        or states.shape[1] != 9
        or len(states) < 4
        or not np.isfinite(states).all()
        or not np.all(states[:, 7:] == states[0, 7:])
        or not isfinite(duration_s)
        or duration_s <= 0
    ):
        raise TimedWeldTourError("finite generated bead, fixed fixture and positive clock required")
    return {
        "times_s": np.linspace(0.0, duration_s, len(states)).tolist(),
        "joint_states_7": states[:, :7].tolist(),
        "fixture": states[0, 7:].tolist(),
        "stages": [{"stage": label, "kind": "weld"}],
        "witnesses": [],
    }


@contextmanager
def preserve_scene(env: Any) -> Iterator[None]:
    """Restore the incoming joint state and revision after fixture-aware work."""
    original = {str(name): float(value) for name, value in env.getState().joints.items()}
    revision = int(env.getRevision())
    try:
        yield
    finally:
        _restore_scene(env, original, revision)
