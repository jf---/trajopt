"""Retain original process collocations while adding observed progress witnesses."""

import numpy as np


class ProgressSampleError(RuntimeError):
    """Progress samples are non-finite, unordered or outside the process leg."""


def progress_sample_times(base, witnesses):
    """Return ordered original and additional times on a single process leg."""
    base = np.asarray(base, dtype=float)
    extra = np.asarray(witnesses, dtype=float)
    if (
        base.ndim != 1
        or len(base) < 2
        or not np.isfinite(base).all()
        or np.any(np.diff(base) <= 0)
        or extra.ndim != 1
        or not np.isfinite(extra).all()
        or np.any(extra < base[0])
        or np.any(extra > base[-1])
    ):
        raise ProgressSampleError("finite ordered samples and in-leg witnesses required")
    return np.unique(np.r_[base, extra])
