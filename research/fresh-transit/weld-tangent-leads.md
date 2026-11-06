# Generated tangent leads: bounded Tour20 experiment

This feasibility experiment preserves the verified generated weld segment and
its modeled clock, while regenerating lead-in and lead-out along extensions of
the bead tangent. It uses the existing common 0.1 m exploratory lead length,
the selected generated rail and fixture, and branch-continuous OPW IK. Each
new lead is a quintic joint fit whose position, velocity and acceleration match
the unchanged cubic bead at the weld event. No stop is inserted in the bead.

For one previously generated Tour20 bead, the
[evidence](tour20-weld-tangent-evidence.json) records 32 IK queries and at
least eight in-limit branches per query. Independent checks found no penetrating
contacts at 1,025 samples per stage, no numerical polynomial position-extrema
excess and no sampled violation of sourced arm/track speeds. The maximum
sampled bead departure from the line was 4.3e-12 m, with modeled feed about
5 mm/s and no sampled reversal. The two join residuals were at most 4.5e-16 in
position, 1.9e-14 per second in velocity and 6.9e-13 per second squared in
acceleration, using the axes' native m/rad units.

The lead fits depart from the ideal generated tangent lines by up to 0.0047 mm
and from uniform lead-time progress by up to 1.50 mm. Their sampled modeled
feed varies from approximately 5 to 79 mm/s against a nominal 50 mm/s; sampled
peak arm jerk is about 35 rad/s³. The lead motion is therefore not a claim of
constant feed or an optimized jerk trajectory. The new lead outer endpoints
differ from the prior candidate and are **not connected** to its old transits.
Lead approach admissibility, manufacturing tolerances, physical controller
timing and continuous collision clearance remain unestablished. This was a
correctness run with concurrent work, not a clean performance benchmark.

The driver needs the previously independently verified fresh candidate, station,
tour index and the research runtime and native libraries. It rejects unsupported
multi-edge beads and records failed checks. With `PY_PATH` and `LIB_PATH` set to
the research modules and matching native libraries, run:

```bash
pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  python research/fresh-transit/weld_tangent_trial.py \
  --station "$STATION" --tour-index "$TOUR_INDEX" --input "$CANDIDATE" \
  --ifopt-library "$IFOPT_LIBRARY" --sqp-library "$SQP_LIBRARY" \
  --weld-stage weld:5 --output "$OUTPUT_DIRECTORY"
```
