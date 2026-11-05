import numpy as np
import pytest
from progress_samples import ProgressSampleError, progress_sample_times


def test_extra_witness_retains_every_original_collocation():
    base = np.linspace(0.0, 1.0, 5)
    times = progress_sample_times(base, [0.875])
    np.testing.assert_array_equal(times, [0.0, 0.25, 0.5, 0.75, 0.875, 1.0])
    assert all(x in times for x in base)


def test_rejects_witness_outside_its_process_leg():
    with pytest.raises(ProgressSampleError):
        progress_sample_times([0.0, 1.0], [1.1])
