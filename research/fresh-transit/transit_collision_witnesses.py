"""Select actual-collision episode witnesses for a failed transit retry."""

import argparse
import json
from pathlib import Path

import numpy as np
from episode_refinement_samples import select_refinement_samples

COLLOCATION_POINTS = 33  # Existing solver's actual-curve collision samples.


class TransitWitnessError(RuntimeError):
    """A failed actual transit or its sampled contacts cannot define witnesses."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contact-audit", "transit-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise TransitWitnessError("output already exists")
    audit = json.loads(args.contact_audit.read_text())
    matches = [
        item
        for item in json.loads(args.transit_results.read_text())["transits"]
        if item["stage"] == audit["stage"]
    ]
    if len(matches) != 1 or matches[0]["status"] != "failed" or not audit["contacts"]:
        raise TransitWitnessError("one failed transit and its sampled contacts required")
    duration = float(matches[0]["optimized_curve"]["duration_s"])
    times = np.linspace(0.0, duration, int(audit["samples"]))
    clearance = np.zeros(len(times))  # Only signs at independently observed contacts are used.
    for contact in audit["contacts"]:
        i = int(contact["sample"])
        if not 0 <= i < len(times) or float(contact["modeled_time_s"]) != times[i]:
            raise TransitWitnessError("sampled contact clock differs from transit")
        clearance[i] = min(clearance[i], float(contact["distance_m"]))
    existing = np.linspace(0.0, duration, COLLOCATION_POINTS + 2)[1:-1]
    witnesses = select_refinement_samples(
        times, clearance, np.zeros((len(times), 9)), existing, np.zeros(9)
    )
    if not witnesses:
        raise TransitWitnessError("no new actual-collision witnesses available")
    args.output.write_text(
        json.dumps({audit["stage"]: witnesses}, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"stage": audit["stage"], "witnesses_s": witnesses}), flush=True)


if __name__ == "__main__":
    main()
