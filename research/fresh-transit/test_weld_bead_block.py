"""A bead block preserves both immediate generated neighbors."""

import pytest
from weld_bead_block import (
    WeldBeadBlockError,
    bead_block_indices,
    representative_bead_block_indices,
)


def test_block_contains_only_failed_bead_and_its_neighbors():
    stages = [{"stage": f"s{i}", "kind": "source_bead"} for i in range(5)]
    assert bead_block_indices(stages, ["s2"]) == (1, 4)


def test_missing_neighbors_or_multiple_targets_fail_loudly():
    stages = [{"stage": f"s{i}", "kind": "source_bead"} for i in range(3)]
    with pytest.raises(WeldBeadBlockError):
        bead_block_indices(stages, ["s0"])
    with pytest.raises(WeldBeadBlockError):
        bead_block_indices(stages, ["s1", "s2"])


def test_uniform_representative_uses_middle_source_bead():
    stages = [
        {"stage": f"s{i}", "kind": "generated_lead" if i in (0, 4) else "source_bead"}
        for i in range(5)
    ]
    assert representative_bead_block_indices(stages) == (1, 4)
