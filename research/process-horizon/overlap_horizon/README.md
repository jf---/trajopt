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
