"""Build a kinematic group against the selected generated fixture state."""

from collections.abc import Sequence
from typing import Any

import numpy as np


class FixtureGroupError(RuntimeError):
    """Fixture names or values cannot define the kinematic-group state."""


def fixture_joint_group(env: Any, manipulator: Any, names: Sequence[str], values: Any) -> Any:
    """Set positioner/tailstock before constructing the group with their transforms."""
    values_array = np.asarray(values, dtype=float)
    if len(names) != 2 or values_array.shape != (2,) or not np.isfinite(values_array).all():
        raise FixtureGroupError("two finite positioner and tailstock values are required")
    env.setState(dict(zip(names, map(float, values_array), strict=True)))
    return env.getJointGroup(manipulator)
