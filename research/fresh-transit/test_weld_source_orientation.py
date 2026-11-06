import numpy as np
import pytest
from weld_source_orientation import build_source_bead_orientation


def test_source_bead_uses_its_own_shortest_arc_and_keeps_lead_curve():
    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    rotations = np.array([rotation(a) for a in (0.0, 0.1, 0.9, 0.5, 0.2)])
    leading = rotation(0.25)
    orientation = build_source_bead_orientation(rotations, [1.0, 2.0, 3.0, 1.0], lambda *_: leading)
    assert orientation(0, 0.5) == pytest.approx(leading)
    assert orientation(3, 0.5) == pytest.approx(leading)
    assert orientation(1, 1.0) == pytest.approx(rotation(0.5))
    assert orientation(2, 1.5) == pytest.approx(rotation(0.7))
