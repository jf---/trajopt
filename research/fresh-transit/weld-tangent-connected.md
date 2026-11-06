# Generated tangent weld and adjacent transits

This experiment connects the regenerated Tour20 tangent leads to **new** OMPL
routes, then refines both routes with native time-aware PIQP splines. The
unchanged generated bead keeps its original modeled clock. The four joins share
modeled joint position, velocity and acceleration; the two far transit
boundaries use the previously generated candidate's modeled derivatives.
Those derivatives are not measured controller states.

The [recorded result](tour20-weld-tangent-connected-evidence.json) covers five
stages: transit, lead-in, bead, lead-out and transit. Both native SQP solves
reported convergence. Independent checking found no penetrating contacts in
1,025 samples per actual stage, no numerical joint-position-extrema excess and
no sampled sourced arm/track speed excess. Maximum sampled speed ratio was
0.822. The four q/qd/qdd join residuals were at most 8.9e-16, 1.9e-14 and
6.9e-13 in the respective axes' m/rad units and modeled seconds.

The bead retained its 8.984 s modeled clock, approximately 5 mm/s feed and
ordered sampled geometry, with maximum position departure 4.4e-12 m. Lead
position departures were approximately 0.0047 mm; their modeled feed varied
from about 5 to 79 mm/s. The lead paths, their speeds and the controller's
approach requirements have **not** been accepted as a manufacturing process.
Collision evidence is finite-sampled, the outer physical derivatives are
unknown, and this local check does not establish whole-tour continuity or
clearance. The correctness run may overlap other work, so its timing is not a
clean benchmark.

The unmodified [lead-only experiment](weld-tangent-leads.md) describes the
source candidate, its bead/lead semantics and local research dependencies.
Using the same research environment, native libraries and saved input, run:

```bash
pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  python research/fresh-transit/weld_tangent_connect_trial.py \
  --station "$STATION" --tour-index "$TOUR_INDEX" --input "$CANDIDATE" \
  --lead-evidence "$LEAD_EVIDENCE" --weld-stage weld:5 \
  --ifopt-library "$IFOPT_LIBRARY" --sqp-library "$SQP_LIBRARY" \
  --output "$OUTPUT_DIRECTORY"
```

This is a one-bead-edge ablation; it does not silently use prior transit
endpoints or claim to handle multiple consecutive bead edges.
