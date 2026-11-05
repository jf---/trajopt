# Fixed-clock probe–transit–probe ablation

This experiment coordinates the scalar progress rates at a generated probe
withdrawal, its following transit and the next generated probe approach. It
retains the three modeled durations, the generated Cartesian IK joint geometry,
probe endpoints and process-corner conditions. A common 3×3 grid of normalized
shared endpoint rates is searched; each transit is reoptimized with the native
fixed-clock quintic jerk objective. This is a discrete bounded search, not a
continuous optimum. Other probe legs, SearchL ordering and fixture events are
outside this neighborhood. The outer q/qd/qdd values are reported against the
existing modeled assembly. SearchL hits are hypothetical, not observed.

Each candidate is independently sampled at 257 points on each process leg and
1,025 points on the actual transit spline for contacts and joint bounds. It
reports source Cartesian departure and sampled progress, q/qd/qdd join gaps,
per-axis dynamics and sampled integral of squared jerk normalized by the sourced
joint speeds. The scalar progress polynomial is checked for reversal at its
interior extrema as well as its endpoints. These checks do not establish
continuous collision freedom, physical jerk limits, controller timing or
manufacturing acceptance.

The runner uses the runtime libraries, Python paths and existing local artifacts
specified in [`../duration_sweep/README.md`](../duration_sweep/README.md). With
those environment variables and paths set, replace the sweep command with:

```bash
python "$NATIVE/research/process-horizon/overlap_horizon/run_horizon.py" \
  --station "$STATION" --original-worktree "$ORIGINAL" \
  --fresh-route "$FRESH_ROUTE" --verified-route "$VERIFIED_ROUTE" \
  --artifacts "$ARTIFACTS" --fixture-policy "$FIXTURE_POLICY" \
  --output "$OUTPUT" --ifopt-library "$NATIVE/build-ifopt/libtrajopt_ifopt.dylib" \
  --sqp-library "$FIXED_SQP/build-sqp/libtrajopt_sqp.dylib" \
  --transit transit:1 --timing-note 'Describe concurrent workload'
```

Add `$NATIVE/research/process-horizon/overlap_horizon` to the beginning of the
`PYTHONPATH` described in that README. Output must not already exist; every
candidate is checkpointed, including explicitly rejected candidates. Loaded
native library paths are verified and recorded.

## First bounded result

The fresh Tour 09 `withdraw:0 → transit:1 → approach:1` experiment kept the
5.28115 s modeled clock. Of the nine candidates, rates `(0.5, 0.5)` had lowest
sampled normalized jerk integral, 4.82784 versus 5.45144 for baseline `(1, 1)`
(11.44% lower). The baseline transit knots matched the separately calculated
fixed-SQP independent transit within 2.29e-14 in native position units. All nine
had no sampled collisions or position-limit excursions. For the selected
candidate, sampled peak arm jerk fell from 15.00 to 14.05 rad/s³, process
position departure was below 1.52e-10 m and no sampled backward progress was
observed. This neighborhood has effectively stationary track, so this result
cannot establish a track-motion improvement. The nine candidate solves and
checks took 2.32 s, with 3.20 s separate scene setup; other host activity was not
characterized. The local untracked `overlap-horizon-transit1/report.json` and
`runtime.json` retain the individual values and actual loaded library paths.

## Whole-tour assembly

`run_whole_tour.py` applies the same rate grid to every eligible, disjoint
withdrawal–transit–approach window, retains all other fixed-SQP transit results,
and checks all 67 stages again with the existing independent actual-curve
verifier. It changes no fixture target, SearchL geometry or stage duration.
Each modified probe leg keeps its outer modeled q/qd/qdd, so its original probe
corners remain in place. The verifier's optional per-leg progress rates preserve
the original default behavior when no override is supplied. Every window and its
full grid are checkpointed; a failed solve or configured sample check remains a
failure. The resulting tour is an exploratory assembly, not a global optimum.

Use the same environment and inputs as above, but use a new `OUTPUT` and replace
`run_horizon.py` with `run_whole_tour.py`. Omit `--transit` and add:

```bash
--baseline-summary "$NATIVE/.superpowers/process-horizon09/fixed-sqp-duration-sweep/global-duration-1/summary.json"
```

The baseline must be a completed fixed-SQP 1× run for the same generated route;
its checks and route preparation were performed separately and are not counted
as the new window-search time. The whole-tour report records those reused inputs,
the scene setup, search plus checks, and the final independent whole-tour checks.

### Whole Tour 09 result

The six eligible windows all completed the common 3×3 grid. All 54 candidates
passed their finite sampled checks; the final 67-stage assembly had no sampled
contacts across 28,547 samples. The modeled tour duration remained 115.505 s,
and the 27 modeled probe-corner stops and hypothetical SearchL hits were
unchanged. Whole-tour maximum q/qd/qdd join differences were 6.50e-15,
4.81e-14 and 1.06e-12 respectively in mixed native units. The existing weld
blends remained the largest measured process departure: 0.918 mm and
7.98e-5 rad; no sampled process progress reversal was observed.

On this modeled clock, sampled sum of arm joint jerk-squared integrals fell
from 20,987.44 to 20,863.10 rad²/s⁵ (0.59%). Whole-tour peak arm jerk remained
125.84 rad/s³, peak track jerk remained 48.01 m/s³, and sampled track jerk-squared
integral remained approximately 3,516.18 m²/s⁵. The window search therefore
improved a measured integral only slightly and did not address the tour's peak
jerk drivers. This is not a global optimum or physical acceptance.

Search and candidate checks took 32.30 s wall / 25.23 s CPU, the separate
whole-tour checks took 7.75 s wall / 5.93 s CPU, and scene setup took 2.95 s.
An unrelated native wheel build ran concurrently, so wall time is contended.
The other transits and their 3.31 s previously measured solve/check baseline
were reused, and fresh-route preparation was not included in this timing. The
untracked `overlap-horizon-whole-tour/summary.json` records all six selections,
per-window reports and the loaded library paths.
