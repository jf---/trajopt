# Fresh Tour 20: opt-in native transit refinement

This is an experimental integration, not a packaged planner. It runs the fresh
Tour 20 graph and route planner, independently checks the generated candidate,
refines its five transit stages with native quintic Ifopt costs and sampled
position and uncapped dynamic collision constraints, and independently checks
the assembled result. Recorded intermediate robot joints are used to recover
process geometry and source timing, never to seed the transit optimization.
The selected generated transit supplies its seven-knot initialization.

The source process durations and the exploratory transit clock are modeled.
Boundary derivatives come from that clock and neighboring joint-linear stages;
the first and last tour velocities are modeled from their transit secants,
not measured. Endpoint accelerations are set to zero. The input probe and weld
states remain unchanged. Their piecewise-linear joins can have velocity jumps:
this experiment does **not** establish full-tour C1, jerk continuity, physical
controller timing, continuous clearance, observed probe hits or executability.
The refiner rejects zero-duration stages and a transit whose fixture moves;
fixture-stage joins and zero-time events need separate integration work.

## Dependencies

This macOS experiment depends on the local research worktrees and libraries;
none are installed by this directory. From the root of this native worktree:

* Build the native spline library and `_quintic_spline` binding using the exact
  instructions in [`trajopt_ifopt/python/README.md`](../../trajopt_ifopt/python/README.md).
* The PIQP/SQP binding and library from the local PR 363 research build are at
  `../piqp-comparison/overlay` and `../piqp-comparison/build-sqp`. The corrected
  dynamic collision binding is at
  `../native-trajopt-spline-dynamic/experimental-python/build`; it uses the
  matching dynamic collision code already present in this native branch.
* Fresh-scene driver and application research modules are from
  `../piqp-fresh-deferred` and `../../docs/superpowers/reviews/2026-11-03-weld-only/probes`.
  Earlier source data, test environment, installed Tesseract libraries and the
  station are in `../../../r62-smooth-windows`. The driver fails if its loaded
  native Ifopt or SQP library differs from the intended paths. This is local
  research infrastructure, not a portable installation recipe.

Run with an unused output path; it refuses to overwrite an existing run:

```bash
RESEARCH=/Users/jelle/Code/CADCAM/tesseract_qt_robodk/.worktrees/r62-smooth-windows
MANIFEST="$RESEARCH/.superpowers/r62-env/pyproject.toml"
PY_PATH="$PWD/research/fresh-transit:$PWD/build-python:$PWD/../native-trajopt-spline-dynamic/experimental-python/build:$PWD/../piqp-comparison/overlay:$PWD/../piqp-fresh-deferred:$PWD/../../docs/superpowers/reviews/2026-11-03-weld-only/probes:$RESEARCH/docs/superpowers/reviews/2026-11-03-weld-only/probes:$RESEARCH/docs/superpowers/reviews/2026-10-23-replace-p2-resolve-routes/probes/oracle:$RESEARCH/.superpowers/r62-master/consumer:$RESEARCH/.superpowers/r62-master:$PWD/../.."
LIB_PATH="$PWD/build-ifopt:$PWD/../piqp-comparison/build-sqp:$RESEARCH/.superpowers/r62-master/install/lib"
pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  python research/fresh-transit/fresh_native_tour20.py \
  --output "$PWD/.superpowers/fresh-native-tour20-new"
PYTEST_XDIST_AUTO_NUM_WORKERS=2 pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  pytest -q -n auto --testmon --testmon-noselect --noconftest \
  research/fresh-transit/test_native_transits.py
```

The output retains the actual spline knot states and boundary derivatives,
modeled clock, QP log and independent assembled report. The assembled verifier
checks 1,025 actual spline samples per transit; its process stages use the
existing 33-sample joint-linear interval policy. Endpoint samples are normalized
by the measured floating-point endpoint residual to match selected graph states;
the underlying spline and the residual are retained. Its reused generic report
text describes the original sampler, so use this paragraph for the actual policy.

## Observed run

On 2026-11-05, an independent fresh run using this path produced five converged
transits and 12 PIQP QPs reporting solved. Planning took 3.080 s, initial checking
1.255 s, refinement including setup 0.509 s, and assembled checking 1.461 s;
measured elapsed time was 6.351 s after scene setup. Scene setup was 3.090 s.
The process was run at lower priority while other work ran; these are local
measurements, not performance guarantees. All 12,820 sampled checks passed;
three probes and one weld retained their original Cartesian geometry. Summed
modeled weighted jerk went from 7,224.71 to 187.60 relative to the interpolated
seven-knot generated input; this is not a comparison against recorded motion.
The maximum sampled speed ratio was 0.782. These finite checks do not establish
continuous clearance or the unsolved whole-tour derivative conditions above.

## Reusable opt-in fresh-tour path

`fresh_native_tour.py` accepts `--tour`, `--station`, `--original` and `--output`;
`fresh_tour_pipeline.plan_and_refine` accepts an already prepared scene and
runs the same fresh planner, verifier and native spline refinement in that
scene. It does not reload a saved candidate or replace the existing Tour 20
trial. The caller explicitly chooses a nine-axis modeled boundary-velocity
policy. The included `modeled_tour_end_secants` policy uses generated endpoint
transit slopes and does not assert physical start or end velocities.

The refinement leaves source process stages intact and verifies the generated
process path using the existing fresh-tour verifier. It independently checks
1,025 actual native spline states per transit for collision, nine-axis bounds
and sampled robot speeds, and evaluates numerical polynomial position extrema.
The existing SQP feasibility band of 0.0001 in native axis units is reported
alongside raw position excursions; it is not a physical tolerance. The output
preserves the clock, generated stages, native spline parameters and per-join
modeled position/velocity differences. A sampled pass does not establish
continuous collision clearance or continuity of the unchanged process stages.

Stationary zero-time events are retained without inventing time. A zero-time
state change or a fixture velocity incompatible with a fixed-fixture transit
fails explicitly. Such a failure is unresolved work, not an accepted tour or a
fallback to the earlier path. The path depends on the same local research
bindings and fresh scene module above. In a fresh Tour 20 smoke run, all five
transits converged and the 5,125 actual spline samples had no collision or
configured-band limit violation; numerical extrema were within the position
bounds. Planning, initial verification, clock preparation, native refinement
and actual-curve checking took 4.009 s after 2.608 s scene setup. This run
shared CPU with another planning worker. The largest modeled process-to-process
velocity jump was 0.0622 rad/s; transit-to-process modeled velocity joins
matched within numerical precision. Tour 20 has no fixture stages. The run
report is `.superpowers/fresh-reusable-20/summary.json` in this native worktree.

Using the variables and environment in the preceding command, run an individual
fresh tour after selecting an unused output directory:

```bash
pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  python research/fresh-transit/fresh_native_tour.py \
  --station /path/to/station-package --original "$RESEARCH" --tour 20 \
  --output "$PWD/.superpowers/fresh-native-tour20-reusable"
PYTEST_XDIST_AUTO_NUM_WORKERS=2 pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  pytest -q -n auto --testmon --testmon-noselect --noconftest \
  research/fresh-transit/test_native_transits.py \
  research/fresh-transit/test_fresh_boundaries.py \
  research/fresh-transit/test_fresh_validation.py \
  research/fresh-transit/test_fresh_joins.py \
  research/fresh-transit/test_fresh_tour_pipeline.py
```
