# Tour 09: fixed-SQP uniform duration experiment

This is an opt-in research runner, not a production planner or a controller-time
model. It replans the 14 transits in a freshly generated 67-stage Tour 09 route,
using one shared scene and the same factor for every transit. It retains the
nine generated probe curves, three exploratory weld blends and eased fixture
chain. The blend departures are measured against the source process geometry;
no manufacturing tolerance is inferred. SearchL hits are assumed, not observed.

The saved baseline supplies modeled join velocities and accelerations, including
approximately 0.96 m/s at the outer track boundaries. These are interpolation
hypotheses, not measured robot velocities. The probe model includes 27 stops.
Fixture event times can shift within the existing fixture-stage clock while its
joint-space chord and total duration are preserved. Zero-duration stages are
unsupported and fail explicitly. There is no recorded intermediate robot-joint
seed in the generated route.

Each transit is optimized with the fixed native SQP and checked at 1,025 actual
spline samples. A separate full-tour pass checks 1,025 transit samples, 257
samples per process or fixture stage, and 65 samples in each localized weld
blend window. It reports sampled contacts, raw nine-axis position-limit excess,
process position/orientation departure and progress, q/qd/qdd join gaps, and
nine-axis speed, acceleration and jerk maxima. Sampled jerk integrals are
numerical approximations. Solver convergence and finite sampling do not prove
continuous collision freedom, jerk continuity, controller feasibility or
manufacturing acceptance. Results are never marked accepted automatically.

## Local inputs and native dependencies

The experiment uses existing local, untracked research artifacts; these are not
included in this source commit. Supply the fresh route and its independent
verification, the process-horizon artifacts including
`tour09-all-weld-blends-assembled.json`, the fixture policy, and the original
station/worktree. The native quintic build and dynamic collision binding must be
built first as described in the parent research documentation. Use the corrected
PIQP-enabled native SQP branch `codex/sqp-accept-improving-step` (commit
`55049ef`), including its `build-sqp/runtime-piqp` runtime library directory.
The runner checks the actual loaded `trajopt_ifopt` and `trajopt_sqp` library
paths before scene setup and fails on a mismatch. Set all paths below to the
local builds and inputs; `OUTPUT` must be absent or empty.

```bash
NATIVE=/absolute/path/to/native-trajopt-quintic
FIXED_SQP=/absolute/path/to/native-trajopt-sqp-accept
INTEGRATION=/absolute/path/to/r62-smooth-windows-main
ORIGINAL=/absolute/path/to/r62-smooth-windows
ENV_MANIFEST="$ORIGINAL/.superpowers/r62-env/pyproject.toml"
STATION=/absolute/path/to/station-cache
FRESH_ROUTE=/absolute/path/to/generated-tour09-route.json
VERIFIED_ROUTE=/absolute/path/to/independently-verified-tour09.json
ARTIFACTS="$NATIVE/.superpowers/process-horizon09"
FIXTURE_POLICY="$NATIVE/.superpowers/fresh-fixture-study/tour09-eased-tracked.json"
OUTPUT=/absolute/path/to/new-sweep-output
DYNAMIC_BINDINGS="$INTEGRATION/.superpowers/native-trajopt-spline-dynamic/experimental-python/build"
PIQP_OVERLAY="$INTEGRATION/.superpowers/piqp-comparison/overlay"

pixi run --as-is --manifest-path "$ENV_MANIFEST" env \
  PYTHONPATH="$NATIVE/research/process-horizon/duration_sweep:$NATIVE/research/process-horizon/experiments:$NATIVE/research/process-horizon:$NATIVE/research/fresh-transit:$NATIVE/build-python:$DYNAMIC_BINDINGS:$PIQP_OVERLAY:$INTEGRATION/.superpowers/piqp-fresh-deferred:$INTEGRATION/docs/superpowers/reviews/2026-11-03-weld-only/probes:$ORIGINAL/docs/superpowers/reviews/2026-10-23-replace-p2-resolve-routes/probes/oracle:$ORIGINAL/.superpowers/r62-master/consumer:$ORIGINAL/.superpowers/r62-bgl-master/experimental-bindings:$ORIGINAL/.superpowers/r62-master:$INTEGRATION" \
  DYLD_LIBRARY_PATH="$NATIVE/build-ifopt:$FIXED_SQP/build-sqp:$FIXED_SQP/build-sqp/runtime-piqp:$ORIGINAL/.superpowers/r62-master/install/lib" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python "$NATIVE/research/process-horizon/duration_sweep/run_uniform_sweep.py" \
  --station "$STATION" --original-worktree "$ORIGINAL" \
  --fresh-route "$FRESH_ROUTE" --verified-route "$VERIFIED_ROUTE" \
  --artifacts "$ARTIFACTS" --fixture-policy "$FIXTURE_POLICY" \
  --output "$OUTPUT" --ifopt-library "$NATIVE/build-ifopt/libtrajopt_ifopt.dylib" \
  --sqp-library "$FIXED_SQP/build-sqp/libtrajopt_sqp.dylib" \
  --timing-note 'Record any concurrent workload here'
```

The default shared factors are 1, 1.5, 2 and 3. Per-transit checkpoints and
per-policy whole-tour reports are retained; a failed solver or configured
sampled check stops the run rather than substituting an older result. The
manifest distinguishes scene setup, transit solving/checks, whole-tour checks,
wall time and process CPU time. The original old-SQP sweep is separate evidence
and must not be presented as a fixed-SQP result.
