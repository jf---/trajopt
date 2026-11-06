"""Checked weld clusters identify each shared generated transit only once."""

import pytest
from weld_transit_parallel import WeldTransitParallelError, expected_transits


def test_checked_groups_include_shared_transit_only_once():
    labels = ["transit:0", "weld:1", "weld:2", "transit:3", "weld:4", "weld:5", "transit:6"]
    recovery = {"verified": {"expanded": {"segments": [{"stage": label} for label in labels]}}}
    batch = {
        "beads": [
            {
                "status": "sampled_exploratory_weld_only",
                "stages": [{"stage": "weld:1"}, {"stage": "weld:2"}],
            },
            {
                "status": "sampled_exploratory_weld_only",
                "stages": [{"stage": "weld:4"}, {"stage": "weld:5"}],
            },
            {"status": "sampled_bead_order_unresolved", "stages": [{"stage": "weld:9"}]},
        ]
    }
    assert expected_transits(recovery, batch) == ["transit:0", "transit:3", "transit:6"]


def test_weld_cluster_without_transit_fails_loudly():
    recovery = {
        "verified": {"expanded": {"segments": [{"stage": "weld:1"}, {"stage": "transit:2"}]}}
    }
    batch = {
        "beads": [{"status": "sampled_exploratory_weld_only", "stages": [{"stage": "weld:1"}]}]
    }
    with pytest.raises(WeldTransitParallelError):
        expected_transits(recovery, batch)
