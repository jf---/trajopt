import pytest
from merge_probe_batches import ProbeMergeError, merge_probe_batches


def test_merge_disjoint_ranges_in_probe_order():
    batches = [
        {
            "tour": 16,
            "probe_start": 2,
            "probe_count": 1,
            "groups": [{"probe": 2, "status": "failed"}],
        },
        {
            "tour": 16,
            "probe_start": 0,
            "probe_count": 2,
            "groups": [{"probe": 0, "status": "ok"}, {"probe": 1, "status": "ok"}],
        },
    ]
    result = merge_probe_batches(batches)
    assert [g["probe"] for g in result["groups"]] == [0, 1, 2]
    assert result["failed_probes"] == [2]


def test_overlap_and_missing_group_fail():
    batch = {
        "tour": 16,
        "probe_start": 0,
        "probe_count": 1,
        "groups": [{"probe": 0, "status": "ok"}],
    }
    with pytest.raises(ProbeMergeError):
        merge_probe_batches([batch, batch])
