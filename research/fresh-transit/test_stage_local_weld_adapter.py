"""Stage-local actual weld curves retain their clocks and interpolation."""

import numpy as np
import pytest
from stage_local_weld_adapter import StageLocalWeldError, stage_local_curves


def record(start, end):
    knots = np.linspace(start, end, 4)
    nodes = np.column_stack([knots**3 + axis for axis in range(7)])
    return {
        "knots_s": knots.tolist(),
        "nodes": nodes.tolist(),
        "v0": np.full(7, 3 * start**2).tolist(),
        "v1": np.full(7, 3 * end**2).tolist(),
    }


def test_stage_local_curves_keep_actual_knots_and_shared_derivatives():
    curves = stage_local_curves([record(0.0, 1.0), record(1.0, 3.0)], [1.0, 2.0])
    for curve, start, end in zip(curves, [0.0, 1.0], [1.0, 3.0], strict=True):
        times = np.linspace(start, end, 19)
        np.testing.assert_allclose(curve(times), np.column_stack([times**3 + i for i in range(7)]))
    for order in range(3):
        np.testing.assert_allclose(curves[0](1.0, order), curves[1](1.0, order), atol=1e-12)


def test_stage_local_curves_reject_wrong_clock_and_axis_count():
    with pytest.raises(StageLocalWeldError):
        stage_local_curves([record(0.0, 1.1)], [1.0])
    bad = record(0.0, 1.0)
    bad["nodes"] = np.zeros((4, 8)).tolist()
    with pytest.raises(StageLocalWeldError):
        stage_local_curves([bad], [1.0])


def test_subset_uses_original_cluster_clock_without_resetting_it():
    curves = stage_local_curves([record(1.0, 3.0)], [2.0], clock_start_s=1.0)
    np.testing.assert_allclose(curves[0](1.5), [1.5**3 + i for i in range(7)])
    with pytest.raises(StageLocalWeldError):
        stage_local_curves([record(1.0, 3.0)], [2.0])


def test_bead_only_window_preserves_generated_leads_and_stage_order():
    from stage_local_weld_adapter import optimization_window

    stages = ["generated_lead", "source_bead", "source_bead", "generated_lead"]
    assert optimization_window(stages, "full") == (0, 4)
    assert optimization_window(stages, "bead") == (1, 3)
    with pytest.raises(StageLocalWeldError):
        optimization_window(["generated_lead", "source_bead"], "bead")
