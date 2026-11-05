import numpy as np
import pytest
from progress_witnesses import (
    ProgressWitnessError,
    ProgressWitnessStalled,
    select_progress_witnesses,
)


def test_selects_ends_of_material_backward_episode():
    times = np.linspace(0, 1, 9)
    progress = [0, 0.2, 0.4, 0.8, 0.7, 0.5, 0.6, 0.8, 1]
    assert select_progress_witnesses(times, progress, [0.0, 1.0], 1e-4) == (0.375, 0.625)


def test_selects_outside_line_extreme():
    times = np.linspace(0, 1, 5)
    progress = [0, 0.5, 1.2, 0.8, 1]
    found = select_progress_witnesses(times, progress, [0.0, 1.0], 1e-4)
    assert 0.5 in found and 0.75 in found


def test_stalls_when_witnesses_already_constrained():
    times = np.linspace(0, 1, 5)
    progress = [0, 0.5, 1.2, 0.8, 1]
    with pytest.raises(ProgressWitnessStalled):
        select_progress_witnesses(times, progress, [0.0, 0.5, 0.75, 1.0], 1e-4)


def test_rejects_bad_clock():
    with pytest.raises(ProgressWitnessError):
        select_progress_witnesses([0.0, 0.0], [0.0, 1.0], [], 1e-4)
