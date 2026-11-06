"""Reconstruct checked generated weld curves and expose unresolved batches."""

from typing import Any

import numpy as np
from actual_stage_adapters import weld_stage
from actual_stage_contract import ActualStage, Seconds
from event_aware_spline_boundaries import STATIONARY_ROUNDOFF_ULPS
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

EXPECTED_COLLISION_SAMPLES = 1025  # Existing finite independent weld-stage check.
GENERATED_IK_ROUNDOFF_ULPS = 32768  # Binary64 generated-IK reconstruction parity.


class WeldBatchAdapterError(RuntimeError):
    """A checked weld curve cannot be connected to its generated tour."""


def adapt_weld_batch(
    rows: list[dict[str, Any]], clock: dict[str, Any], batch: dict[str, Any], tour: int
) -> tuple[dict[int, tuple[ActualStage, Any]], list[dict[str, Any]]]:
    """Reconstruct checked curves; preserve failed groups as explicit unresolved data.

    States use track metres, six arm radians, positioner radians and tailstock
    metres; stage clocks are modeled seconds. This does not approve a process
    deviation or establish real controller timing.
    """
    if batch["tour"] != tour or len(rows) != len(clock["stages"]):
        raise WeldBatchAdapterError("tour or modeled clock mismatch")
    labels = [row["stage"] for row in rows]
    if len(set(labels)) != len(labels):
        raise WeldBatchAdapterError("generated stage labels are not unique")
    lookup = {label: i for i, label in enumerate(labels)}
    stages: dict[int, tuple[ActualStage, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for bead in batch["beads"]:
        if bead["status"] != "sampled_exploratory_weld_only":
            unresolved.append({"source_segment": bead["source_segment"], "status": bead["status"]})
            continue
        if any(
            not stage.get("sampled_speeds_pass")
            or stage.get("collision_and_position", {}).get("sample_count")
            != EXPECTED_COLLISION_SAMPLES
            or stage.get("collision_and_position", {}).get("penetrating_samples") != 0
            for stage in bead["stages"]
        ):
            raise WeldBatchAdapterError(
                "checked weld stage lacks its sampled collision or speed result"
            )
        names = [stage["stage"] for stage in bead["stages"]]
        if any(name not in lookup for name in names):
            raise WeldBatchAdapterError("weld stage is missing from the generated tour")
        indices = [lookup[name] for name in names]
        if not indices or indices != list(range(indices[0], indices[0] + len(indices))):
            raise WeldBatchAdapterError("weld stages are not contiguous")
        if any(index in stages for index in indices):
            raise WeldBatchAdapterError("weld stages overlap")
        durations = np.asarray([clock["stages"][i]["duration_s"] for i in indices], dtype=float)
        times = np.asarray(bead["times_s"], dtype=float)
        nodes = np.asarray(bead["joint_states_7"], dtype=float)
        fixture = np.asarray(bead["fixture"], dtype=float)
        if (
            any(clock["stages"][i]["stage"] != labels[i] for i in indices)
            or not np.isfinite(durations).all()
            or np.any(durations <= 0)
            or times.ndim != 1
            or len(times) < 2
            or nodes.shape != (len(times), 7)
            or fixture.shape != (2,)
            or not np.isfinite(times).all()
            or not np.isfinite(nodes).all()
            or not np.isfinite(fixture).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise WeldBatchAdapterError("invalid modeled weld clock, nodes or fixture")
        edges = np.r_[0.0, np.cumsum(durations)]
        band = (
            GENERATED_IK_ROUNDOFF_ULPS
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(nodes))))
        )
        time_band = STATIONARY_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, float(edges[-1]))
        if abs(times[0]) > time_band or abs(times[-1] - edges[-1]) > time_band:
            raise WeldBatchAdapterError("weld curve and modeled clock differ")
        curve = CubicSpline(times, nodes, axis=0)
        for index, knot in ((indices[0], nodes[0]), (indices[-1], nodes[-1])):
            source = np.asarray(rows[index]["states"], dtype=float)
            if source.ndim != 2 or source.shape[1] != 9 or len(source) < 2:
                raise WeldBatchAdapterError("generated weld states invalid")
            endpoint = source[0] if index == indices[0] else source[-1]
            delta = endpoint - np.r_[knot, fixture]
            if np.max(np.abs(delta)) > band:
                raise WeldBatchAdapterError(
                    f"{labels[index]} generated weld boundary changed: {delta.tolist()}"
                )
        for j, index in enumerate(indices):
            source = np.asarray(rows[index]["states"], dtype=float)
            if (
                source.ndim != 2
                or source.shape[1] != 9
                or np.any(np.abs(source[:, 7:] - fixture) > band)
            ):
                raise WeldBatchAdapterError("weld fixture changed")
            duration = float(durations[j])
            actual = weld_stage(labels[index], duration, curve, float(edges[j]), tuple(fixture))
            samples = curve(np.linspace(edges[j], edges[j + 1], len(source)))
            nine = np.column_stack((samples, np.tile(fixture, (len(samples), 1))))
            stages[index] = (actual, nine)
    return stages, unresolved


def weld_neighbor_indices(
    rows: list[dict[str, Any]], stages: dict[int, tuple[ActualStage, Any]]
) -> set[int]:
    """Return each generated neighboring transit once for checked weld clusters."""
    indices = set(stages)
    transits: set[int] = set()
    for index in indices:
        for direction in (-1, 1):
            adjacent = index + direction
            if adjacent in indices:
                continue
            if not 0 <= adjacent < len(rows) or not rows[adjacent]["stage"].startswith("transit:"):
                raise WeldBatchAdapterError("checked weld cluster lacks an adjacent transit")
            transits.add(adjacent)
    return transits


def canonicalize_weld_transit_endpoints(
    rows: list[dict[str, Any]], stages: dict[int, tuple[ActualStage, Any]]
) -> list[dict[str, Any]]:
    """Share each checked cubic endpoint with its adjacent generated transit.

    Only binary64 generated-IK reconstruction differences are eligible. The
    transit is subsequently reoptimized and independently checked. Fixture
    axes are retained; source-to-generated nine-axis deltas are reported.
    """
    indices = set(stages)
    pending: list[tuple[int, int, Any, dict[str, Any]]] = []
    for index in sorted(indices):
        stage = stages[index][0]
        for direction in (-1, 1):
            adjacent = index + direction
            if adjacent in indices:
                continue
            if not 0 <= adjacent < len(rows) or not rows[adjacent]["stage"].startswith("transit:"):
                raise WeldBatchAdapterError("checked weld cluster lacks an adjacent transit")
            q = np.asarray(rows[adjacent]["states"], dtype=float).copy()
            if q.ndim != 2 or q.shape[1] != 9 or len(q) < 2 or not np.isfinite(q).all():
                raise WeldBatchAdapterError("finite nine-axis generated transit required")
            endpoint = -1 if direction == -1 else 0
            modeled: Any = np.asarray(
                tuple(stage.evaluate(Seconds(0.0) if direction == -1 else stage.duration).position),
                dtype=float,
            )
            delta = modeled - q[endpoint]
            band = (
                GENERATED_IK_ROUNDOFF_ULPS
                * np.finfo(float).eps
                * max(1.0, float(np.max(np.abs(q))), float(np.max(np.abs(modeled))))
            )
            if np.max(np.abs(delta)) > band or np.any(delta[7:] != 0):
                raise WeldBatchAdapterError("generated weld/transit endpoint or fixture differs")
            q[endpoint, :7] = modeled[:7]
            pending.append(
                (
                    adjacent,
                    endpoint,
                    q[endpoint].copy(),
                    {
                        "transit": rows[adjacent]["stage"],
                        "weld": stage.label,
                        "raw_source_to_cubic_delta": delta.tolist(),
                    },
                )
            )
    for adjacent, endpoint, value, _ in pending:
        q = np.asarray(rows[adjacent]["states"], dtype=float).copy()
        q[endpoint] = value
        rows[adjacent] = {**rows[adjacent], "states": q.tolist()}
    return [entry for _, _, _, entry in pending]
