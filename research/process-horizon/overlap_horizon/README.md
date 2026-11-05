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

## Three-probe rail and clock exploration

A separate bounded Tour 09 exploration regenerated OPW branches for probes 3,
4 and 5 at a common 2.6 m track position. It connected their generated probe
macros between the existing fixture exit and weld entry using four native
transit optimizations. Each affected transit used the same speed-derived rule:
measure the solved spline's per-axis numerical velocity extrema, multiply its
duration by the maximum sourced-speed ratio if necessary, and reoptimize with
the original shared q/qd/qdd. Numerical polynomial roots are not an interval
proof. The solver's actual spline and the independent polynomial representation
are compared at the reported extrema. This is a candidate-specific exploration,
not a universal rail-selection policy or a continuous global optimum.

The 16-stage horizon passed its finite actual-curve checks, and splicing it into
the existing 67-stage assembly yielded no sampled contacts in 28,547 samples.
The unchanged fixture and weld boundaries retained their modeled q/qd/qdd; the
original probe geometry, hypothetical SearchL hits and 27 modeled corner stops
remain. The maximum sampled process departure remained 0.918 mm and 7.98e-5 rad
from the earlier weld blend, with no sampled backward progress. No physical,
controller or manufacturing acceptance is claimed.

The whole-tour sampled speed-normalized jerk-squared integral fell from 2,302.24
to 2,049.67 (10.97%), while modeled duration rose from 115.50478 to 115.58811 s.
The sampled track jerk-squared integral fell from 3,516.18 to 2,847.11 m²/s⁵,
but **peak** track jerk rose from 48.01 to 53.09 m/s³. Peak track acceleration
rose from 4.62 to 5.32 m/s² and peak track speed from 1.5 to 1.8 m/s; peak arm
jerk remained 125.84 rad/s³. The four changed transits have numerical polynomial
speed-extrema checks; other stage speeds are sampled only. The source fixtures
and generated probe artifacts in the local untracked
`three-probe-speed-policy-horizon/` and `three-probe-whole-tour/` directories
preserve the full candidate and results. Scene/build contention was not fully
characterized, and the graph retained one generated process path per sampled
endpoint pair, so these measurements do not characterize production performance.

### Same-policy 2× comparison

The same bounded 2.6 m generated candidate was compared under the existing
uniform 2× transit-duration policy. All four changed transit splines were
reoptimized against their actual process, fixture and weld q/qd/qdd boundaries;
the other verified 2× baseline transits were reused. The independent whole-tour
check again covered 67 stages and 28,547 samples with no contacts or configured
position-bound excess. Modeled duration was 129.64338 s for both paths, and the
fixture and process clocks were unchanged. The four changed transits also had
numerical polynomial speed-extrema checks.

The sampled speed-normalized jerk-squared integral fell from 132.62 to 128.99
(2.73%), and the track jerk-squared integral fell from 117.39 to 104.80 m²/s⁵
(10.72%). However, sampled **peak track jerk rose** from 6.03 to 7.87 m/s³ and
peak acceleration from 1.155 to 1.443 m/s²; peak track speed remained 0.96 m/s
and peak arm jerk remained 40.45 rad/s³. Process departure and all other
physical-acceptance limitations above are unchanged. Thus the tested objective
improved the integrated measure while worsening this peak. The 2× comparison is
recorded in the untracked `three-probe-whole-tour-2x/summary.json`; it reused
previously computed baseline and generated process artifacts, and concurrent
host activity was not characterized.
