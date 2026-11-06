# Tour 16: partial coordinated assembly

The fresh generated candidate is assembled with 223 coordinated probe groups,
69 fixture stages in 36 ordered chains, 311 checked transit curves and one
checked lead. Every shared transit is solved once after its neighboring probe
boundaries are available; 36 fixture-adjacent transits are reused. This does not
use recorded intermediate robot joints as a seed.

The 1,548-stage assembly is incomplete: `transit:0` has an unknown physical
start velocity and remains unresolved; `transit:655` and 273 other weld or
lead stages remain generated joint-linear. Among 1,547 modeled joins, the
largest velocity gaps are 0.558 m/s for track and 2.329 rad/s for an arm axis
at `transit:0` → `approach:0`, and 1.437 rad/s for an arm axis at `weld:654` →
`transit:655`. There are also unrefined internal joint-linear corners. Join
counts use a numerical parity band and are not themselves physical acceptance
criteria.

Independent finite checks covered 1,304,825 sampled configurations across the
checked curves without detected penetration. The remaining stages were not
rechecked in this experiment; this is not a full-tour or continuous-clearance
claim. Transit position extrema are numerically evaluated; their speed checks
are sampled. The largest sampled SearchL departure from its target line is
0.041 mm, and its largest backward displacement is 0.049 mm. The largest
SearchL clock-target difference is 0.366 mm. These are reported measurements,
not manufacturing tolerances. The hit at the search limit is hypothetical.

The same 1.5 factor changes the modeled fixture time from 37.775 to 56.662 s.
Fixture dynamics and physical start/end derivatives remain unknown. Neither the
modeled clock nor process-feed assumptions represent measured controller time.

Four worker-owned scenes processed 276 distinct probe-adjacent transits: 275
passed the stated checks and `transit:0` remained unresolved. Their elapsed
wall time was 26.483 s including setup, with a maximum per-worker refinement
span of 22.274 s; another workload overlapped. This is not an uncontended speed
comparison. The [compact evidence](tour16-probe-transit-assembly.json) includes
per-group and per-transit outcomes, join gaps, worker metrics and the local
untracked source paths.

The scripts are research-only and depend on the local scene and corrected native
libraries described in [the runtime instructions](README.md). Given the
independently generated SearchL batch, fixture checkpoint, fresh recovery, station
and matching native libraries, the transit step is reproduced with:

```bash
pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  python research/fresh-transit/probe_transit_parallel.py \
  --station "$STATION" --tour-index "$TOUR_INDEX" --recovery "$RECOVERY" \
  --resume-from "$FIXTURE_CHECKPOINT" --probe-batch "$PROBE_BATCH" \
  --ifopt-library "$IFOPT_LIBRARY" --sqp-library "$SQP_LIBRARY" \
  --fixture-duration-factor 1.5 --workers 4 --output "$OUTPUT_DIRECTORY"
```

The combined `results.json` is then supplied with `--probe-transit-results`
to `fixture_transit_tour_trial.py`, together with `--checked-neighborhood` for
the separately checked lead. Every input and the unused output directory must
be explicitly supplied; missing inputs or failed workers are not silently
replaced. Local untracked trial outputs are evidence, not packaged dependencies.
