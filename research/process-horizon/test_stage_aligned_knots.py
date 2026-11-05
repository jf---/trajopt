import numpy as np
import pytest
from stage_aligned_knots import KnotClockError, stage_aligned_knots


def test_keeps_budget_and_includes_every_process_boundary():
    edges = np.array([0.0, 1.83166, 3.73166, 4.11166, 5.94332])
    times = stage_aligned_knots(edges, 13)
    assert len(times) == 13
    assert np.all(np.diff(times) > 0)
    for edge in edges:
        assert edge in times
    assert sum((times >= edges[2]) & (times <= edges[3])) >= 2


def test_rejects_missing_intervals_or_nonincreasing_boundaries():
    with pytest.raises(KnotClockError):
        stage_aligned_knots([0, 1, 2, 3], 3)
    with pytest.raises(KnotClockError):
        stage_aligned_knots([0, 1, 1], 5)
