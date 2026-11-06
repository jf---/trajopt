"""Contracts for tangent leads adjoining the unchanged weld."""

import numpy as np
import pytest
from weld_tangent_leads import WeldLeadError, fit_lead, tangent_lead_targets


def test_tangent_lead_targets_leave_bead_endpoints_and_orientations_unchanged():
    a, b = np.eye(4), np.eye(4)
    a[:3, 3], b[:3, 3] = [1, 2, 3], [1, 4, 3]
    lead_in, lead_out = tangent_lead_targets(a, b, 0.1, 5)
    np.testing.assert_allclose(lead_in[0, :3, 3], [1, 1.9, 3])
    np.testing.assert_allclose(lead_in[-1], a)
    np.testing.assert_allclose(lead_out[0], b)
    np.testing.assert_allclose(lead_out[-1, :3, 3], [1, 4.1, 3])
    np.testing.assert_allclose(lead_in[:, :3, :3], np.broadcast_to(np.eye(3), (5, 3, 3)))


def test_fit_matches_supplied_weld_derivatives_without_changing_weld_endpoint():
    times = np.linspace(0, 1, 8)
    samples = np.column_stack([times**2] * 7)
    curve = fit_lead(times, samples, (1.0, 2.0), (3.0, 4.0))
    np.testing.assert_allclose(curve(times), samples, atol=1e-12)
    np.testing.assert_allclose(curve(0.0, 1), np.ones(7), atol=1e-12)
    np.testing.assert_allclose(curve(0.0, 2), np.ones(7) * 2, atol=1e-11)
    np.testing.assert_allclose(curve(1.0, 1), np.ones(7) * 3, atol=1e-12)
    np.testing.assert_allclose(curve(1.0, 2), np.ones(7) * 4, atol=1e-11)


def test_degenerate_bead_is_refused():
    with pytest.raises(WeldLeadError, match="nonzero"):
        tangent_lead_targets(np.eye(4), np.eye(4), 0.1, 5)
