"""Prepare fresh selected stages and the existing modeled clock in memory."""

import numpy as np
from expand_multi_episode import expand_multi_episode
from fixture_source_clock import fixture_source_durations
from multi_episode_graph import build_multi_episode_graph
from probe_episodes import partition_process_tour
from probe_process_inputs import extract_process_tour
from shortcut_source_durations import extract_shortcut_source_durations
from shortcut_tour_clock import build_shortcut_clock
from weld_only_inputs import extract_tour

from tesseract_qt_py.core.program.program_wire import ExecutionRole


class FreshPreparationError(RuntimeError):
    """Generated and independently checked stages or speed inputs differ."""


def prepare_fresh(env, program, source, tours, home, names, raw, metadata, tour, record, checked):
    """Derive a clock from the checked generated path without loading a saved candidate."""
    if checked.get("status") != "independently_sampled_tour" or checked.get("tour") != tour:
        raise FreshPreparationError("checked candidate is missing or belongs to another tour")
    speeds = []
    for axis in names[:7]:
        data = metadata[axis]
        if (
            data["speed_source"] not in ("datasheet", "cell_declared")
            or float(data["velocity"]) <= 0
        ):
            raise FreshPreparationError(f"missing sourced velocity for {axis}")
        speeds.append(float(data["velocity"]))
    task = extract_tour(source, tours, tour, "workpiece_link", home=home)
    process = extract_process_tour(program, task, probe_metadata_frame="workpiece_link")
    taken = [
        s
        for s in program.body.segments
        if s.execution.role is ExecutionRole.TAKEN and s.execution.tour == tour
    ]
    if not taken:
        raise FreshPreparationError("tour has no source segments")
    dummy = np.zeros((1, 7))
    graph = build_multi_episode_graph(
        partition_process_tour(process),
        np.asarray(taken[0].rows[0]),
        np.asarray(taken[-1].rows[-1]),
        lambda _: dummy,
        lambda weld: tuple(dummy for _ in range(len(weld.poses) + 2)),
        lambda _: dummy,
    )
    expanded = expand_multi_episode(graph, record["result"])
    if checked["expanded"]["segments"] != expanded["segments"]:
        raise FreshPreparationError("generated and independently checked states differ")
    durations = extract_shortcut_source_durations(
        env, program, process, graph, expanded, tuple(names)
    )
    durations.update(fixture_source_durations(raw, checked))
    segments = expanded["segments"]
    return segments, build_shortcut_clock(segments, durations, speeds), tuple(speeds)
