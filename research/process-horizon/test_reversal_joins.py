"""Only coincident, opposing nondegenerate process lines require a reversal."""

import numpy as np
import pytest
from reversal_joins import ReversalGeometryError, opposing_joins


def test_only_shared_opposing_lines():
    legs = [
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    ]
    assert opposing_joins(legs) == [0]
    legs[1] = (np.array([1.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert opposing_joins(legs) == []


def test_reject_degenerate_line():
    with pytest.raises(ReversalGeometryError):
        opposing_joins([(np.zeros(3), np.zeros(3))])
