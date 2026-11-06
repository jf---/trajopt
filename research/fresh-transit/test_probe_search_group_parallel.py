import pytest
from probe_search_group_parallel import ProbeParallelError, split_ranges


def test_worker_ranges_cover_each_probe_once_without_empty_worker():
    ranges = split_ranges(8, 215, 4)
    covered = [index for start, count in ranges for index in range(start, start + count)]
    assert covered == list(range(8, 223))
    assert max(count for _, count in ranges) - min(count for _, count in ranges) <= 1


def test_more_workers_than_probes_fails():
    with pytest.raises(ProbeParallelError):
        split_ranges(0, 2, 3)
