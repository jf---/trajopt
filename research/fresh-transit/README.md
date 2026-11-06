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

## Opt-in fixture-chain clock experiment

`fixture_chain_clock` constructs a quintic progress clock along one unchanged,
monotone, straight nine-axis fixture chain. It retains the ordered fixture
waypoints and stops at the chain endpoints, without adding stops at intermediate
waypoints. It rejects non-collinear paths, reversals and stationary fixture
stages, since their event or dwell semantics need a separate clock. The caller
must explicitly opt in and supply the modeled total duration. `fixture_chain_trial`
uses this clock to give adjacent native transit splines consistent modeled
fixture endpoint velocities, and independently samples both kinds of curve.
The fixed process stages and original process verification remain separate.

The fresh Tour 09 candidate has five ordered fixture stages, all a single
positioner sweep from the same source transit segment, with no recorded
`discontinuity_indices`. The same modeled fixture duration of **0.61181 s** was
retained while intermediate waypoint times moved; the modeled positioner peak
speed is **4.811 rad/s**, compared with a largest source-clock segment secant
of **5.186 rad/s**. Positioner speed and acceleration limits are unsourced, so
these are not machine-limit checks or physical timing claims. The source
segment does not establish controller interpolation or prove that eliminating
intermediate stops is appropriate for execution.

On this fresh candidate, the opt-in trial returned 14 converged native transit
splines. The 14,350 actual transit samples and 1,025 actual fixture samples had
no sampled collisions or configured-band nine-axis limit violations; numerical
transit position extrema were also checked. Transit/fixture and internal fixture
modeled velocities match at the joins. The unchanged process stages still have
modeled velocity jumps up to 0.0818 rad/s, so whole-tour smoothness is not
established. The isolated trial took **3.51 s** including its checks after a
separately prepared scene and previously generated candidate. Its local
untracked evidence is `.superpowers/fresh-fixture-study/tour09-eased-tracked.json`
and the PIQP log is next to it; this is not an all-tour or cold-planning result.

## Actual-stage and event contract

`actual_stage_contract.py` is an additive interface for nine-axis modeled curves.
Each positive-duration stage supplies its actual position and known or unknown
velocity and acceleration. Stationary zero-time events retain their names and
order; they do not imply that the robot stops. A continuity gate requires
caller-supplied axis-specific bands and rejects unknown derivatives. It does not
choose physical tolerances or infer controller boundary conditions.

`actual_stage_adapters.py` exposes the native quintic transit, generated probe
progress, existing fixture easing and generated weld curves through this
contract. The opt-in Tour09 sampled verifier also checks each adapter against
its existing evaluation at both endpoints and an interior sample. The adapters
must not change default verifier behavior. Process rates that would reverse the
modeled progress are rejected rather than clipped into a misleading derivative.

The generated Tour09 2× assembly was rechecked using the β=0.1 transit curves:
67 stages, 28,547 sampled states, no detected penetrating contacts and the
configured joint-position and sourced arm/track speed checks passed. All 66
internal joins had modeled position, velocity and acceleration derivatives;
endpoint adapter parity and sampled checks passed. Outer physical derivatives
remain unknown. These are modeled finite-sample checks on one tour; they do not
establish continuous collision clearance, controller timing, all-tour coverage
or manufacturing acceptance. The reusable inputs and detailed local report are
under the ignored `.superpowers/process-horizon09/adapter-stage-parity-checked/`.

The compact [stage-parity report](tour09-actual-stage-parity.json) records the
result. The experiment driver is
`research/process-horizon/duration_sweep/check_actual_stage_adapters.py`. It
requires explicit station, original worktree, fresh-route, verified-route,
process artifacts, fixture policy, generated transit summary, native Ifopt and
corrected SQP library paths. Run it in the documented experimental Pixi runtime
with the `fresh-transit`, `process-horizon/experiments`,
`process-horizon/duration_sweep` and native binding modules on `PYTHONPATH`;
pass `--help` for its complete CLI. Those local development libraries and
artifact paths are research dependencies, not installed product components.

## Fresh product and free outer acceleration

The opt-in `fresh_product_jerk_trial.py` consumes an independently verified
fresh product candidate in its retained worker scene. It uses one explicit
β=0.1 peak weight, 1 s reference time, 1e−9 SQP absolute improvement and at
most 12 adaptive witness rounds. Unknown outer acceleration is a free native
optimization variable, not a controller measurement. Neighboring process
acceleration comes from the explicit joint-linear interpolation model; a fixed
fixture transit rejects an incompatible moving fixture boundary. Every transit
is checked on its actual quintic; the independently verified original process
and fixture geometry remains unchanged. The acceptance gate stays unresolved
at unknown physical boundaries and sharp joint-linear interpolation corners.

The [fresh Tour20 evidence](tour20-fresh-free-acceleration.json) records five
actual spline checks and the modeled optimized endpoint accelerations. The
225 reported process corners are undefined derivatives of the chosen
piecewise-linear representation, not a count of physical robot defects.
A separate fresh Tour09 attempt failed at a fixture/transit velocity mismatch;
it was not accepted. The driver requires the installed research wheel plus the
native Ifopt/SQP libraries explicitly and verifies their loaded paths. Its
`--help` describes the station, tour-index, output and runtime inputs.

For a reproducible local run, set the station, tour-index, installed wheel
module, native Ifopt and corrected SQP library paths explicitly. The runtime
needs the existing research modules and the dynamic spline collision binding in
`PY_PATH`, and the corresponding native library directories in `LIB_PATH`:

```bash
pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PY_PATH" DYLD_LIBRARY_PATH="$LIB_PATH" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python research/fresh-transit/fresh_product_jerk_trial.py \
  --station "$STATION" --tour-index "$TOUR_INDEX" \
  --expected-wheel-module "$WHEEL_MODULE" --expected-ifopt-library "$IFOPT_LIBRARY" \
  --expected-sqp-library "$SQP_LIBRARY" --tour 20 --tour 9 \
  --output "$OUTPUT_DIRECTORY"
```

The inputs are development artifacts, not a packaged product; the driver rejects
an already existing output directory and verifies the loaded libraries.
