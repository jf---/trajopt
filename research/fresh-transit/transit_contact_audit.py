"""Audit every finite-sample contact on an actual generated transit curve."""

import argparse
import json
from pathlib import Path

import _quintic_spline as native
import numpy as np
import tesseract_robotics.trajopt_sqp  # noqa: F401
from fresh_product_scene import initialize_scene, snapshot
from runtime_libraries import require_loaded_libraries

from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

SAMPLES = 1025  # Match the independent transit validation grid.


class TransitContactAuditError(RuntimeError):
    """A saved transit or scene cannot be inspected."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "station",
        "tour-index",
        "recovery",
        "input",
        "output",
        "ifopt-library",
        "sqp-library",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    require_loaded_libraries(
        [args.ifopt_library.resolve(strict=True), args.sqp_library.resolve(strict=True)]
    )
    if args.output.exists():
        raise TransitContactAuditError("output already exists")
    args.output.mkdir(parents=True)
    scene = initialize_scene(args.station, args.tour_index, args.output)
    original = snapshot(scene)
    env, _, _, _, names, _, _, _, _, _, _, _ = scene["inputs"]
    recovery = json.loads(args.recovery.read_text())
    labels = [r["stage"] for r in recovery["verified"]["expanded"]["segments"]]
    index = labels.index(args.stage)
    data = json.loads(args.input.read_text())
    matches = [t for t in data["transits"] if t["stage"] == args.stage]
    if len(matches) != 1 or "optimized_curve" not in matches[0]:
        raise TransitContactAuditError("one saved actual transit curve required")
    curve = matches[0]["optimized_curve"]
    model = native.QuinticSplineModel.build(
        curve["knots_s"], *(np.asarray(curve[k]) for k in ("v0", "a0", "v1", "a1"))
    )
    times = np.linspace(0.0, float(curve["duration_s"]), SAMPLES)
    q = model.evaluate(times, np.asarray(curve["nodes"]), 0)[0]
    checker = StateChecker.build(env, tuple(names), {}, PairMargins.uniform(Meters(0)))
    contacts = []
    for i, state in enumerate(q):
        for contact in checker.contacts(np.r_[state, curve["fixture"]]):
            contacts.append(
                {
                    "sample": i,
                    "modeled_time_s": float(times[i]),
                    "links": list(contact.links),
                    "distance_m": float(contact.distance_m),
                    "margin_m": float(contact.margin_m),
                }
            )
    if snapshot(scene) != original:
        raise TransitContactAuditError("scene state changed")
    result = {
        "stage": args.stage,
        "previous_stage": labels[index - 1] if index else None,
        "next_stage": labels[index + 1] if index + 1 < len(labels) else None,
        "samples": SAMPLES,
        "contacts": contacts,
        "scope": "Sampled generated transit under zero contact margins; physical significance and continuous clearance are not asserted.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "stage": args.stage,
                "sampled_contacts": len(contacts),
                "pairs": sorted({tuple(c["links"]) for c in contacts}),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
