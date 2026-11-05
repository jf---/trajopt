# Process-horizon research

This opt-in experiment follows one freshly generated Tour 09 probe and its two adjacent transits. It derives the process poses from the input program; the robot configurations used by the branch continuation come from the generated route and OPW IK. SearchL ends at a **hypothetical no-hit limit**. It does not model measured contact or controller correction. The fixture remains fixed within this neighborhood.

The driver compares four process stages: approach, SearchL, retract and withdrawal. It follows the Cartesian target by IK at 17 points per leg, chooses the branch nearest the preceding generated state, interpolates the resulting joint curves and applies a modeled scalar clock. At the three detected incompatible process-twist joins, this particular interpolation requires zero TCP velocity for exact first-order continuity. The outer process velocities remain nonzero. Both neighboring transits use native quintic jerk costs, with their boundary velocities and accelerations matched to the process curve. The next unmodified approach uses a modeled joint-linear secant and zero acceleration; this is not controller feedback.

The retained evidence in `evidence/` is one run, not an all-tour result. It reports 3,078 sampled states, no sampled penetration or joint-limit excess, and matched position, velocity and acceleration at the two process/transit joins. These are finite-sample checks, not continuous collision certification. The cubic process curves are twice continuously differentiable within each leg, but jerk may jump at their internal knots. Their clock, acceleration and jerk are modeled; no controller timing or physical dynamic limits have been established. The source-derived Cartesian interpolation and hypothetical SearchL endpoint remain assumptions to validate against the intended controller semantics.

The native Cartesian line and progress constraints and the fixed-generated-join linear velocity constraint are separate experiments. The latter is valid only when the caller fixes the join configuration used to compute its progress gradient. The earlier global joint-quintic experiments did not establish continuous Cartesian fidelity; their failed iterates remain local research artifacts, not accepted paths.

## Reproduction

The driver requires the project Pixi environment, an independently verified fresh Tour 09 route and the matching station package. It also needs the locally built native spline/Ifopt library, the uncapped dynamic collision binding, the PIQP-enabled SQP library and the corresponding nanobind overlay. These are research dependencies, not vendored binaries. The input route must be generated without recorded intermediate robot joint seeds, and its verification JSON must describe the same expanded stages; `prepare_fresh` checks that relationship. Existing recorded start/end cell states and source process geometry are used as documented by the fresh planner.

Build the native library and Python binding using the configured build directories, then run the focused contracts:

```bash
cmake --build "$NATIVE/build-ifopt" -j 4
cmake --build "$NATIVE/build-python" -j 4
pixi run --as-is --manifest-path "$ENV_MANIFEST" env \
  TESSERACT_RESOURCE_PATH="$ORIGINAL/.superpowers/r62-master/install/share" \
  ctest --test-dir "$NATIVE/build-ifopt" -R 'SplineCartesian(Line|Progress)Constraint' --output-on-failure
```

Set `NATIVE` to this repository, `INTEGRATION` and `ORIGINAL` to the isolated integration and original project worktrees, and `ENV_MANIFEST` to the project Pixi environment manifest. Set `STATION`, `FRESH_ROUTE`, `VERIFIED_ROUTE` and `OUTPUT` to the station directory, generated route JSON, matching verified route JSON and a writable durable experiment directory. `DYNAMIC` is the directory containing `_spline_dynamic`; `PIQP_OVERLAY` is the matching nanobind overlay; `PIQP_LIB` is the directory containing the PIQP-enabled SQP library. The process validates the actual loaded Ifopt and SQP library paths before scene initialization. On macOS:

```bash
pixi run --as-is --manifest-path "$ENV_MANIFEST" env \
  PYTHONPATH="$NATIVE/research/process-horizon:$NATIVE/research/fresh-transit:$NATIVE/build-python:$DYNAMIC:$PIQP_OVERLAY:$INTEGRATION/.superpowers/piqp-fresh-deferred:$INTEGRATION/docs/superpowers/reviews/2026-11-03-weld-only/probes:$ORIGINAL/docs/superpowers/reviews/2026-11-03-weld-only/probes:$ORIGINAL/docs/superpowers/reviews/2026-10-23-replace-p2-resolve-routes/probes/oracle:$ORIGINAL/.superpowers/r62-master/consumer:$ORIGINAL/.superpowers/r62-bgl-master/experimental-bindings:$ORIGINAL/.superpowers/r62-master:$INTEGRATION" \
  DYLD_LIBRARY_PATH="$NATIVE/build-ifopt:$PIQP_LIB:$ORIGINAL/.superpowers/r62-master/install/lib" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python "$NATIVE/research/process-horizon/experiments/run_probe_neighborhood.py" \
    --station "$STATION" --original-worktree "$ORIGINAL" \
    --fresh-route "$FRESH_ROUTE" --verified-route "$VERIFIED_ROUTE" --output "$OUTPUT" \
    --ifopt-library "$NATIVE/build-ifopt/libtrajopt_ifopt.dylib" \
    --sqp-library "$PIQP_LIB/libtrajopt_sqp.dylib"
```

Pass `--retain-scene` to keep the environment available for further experiments. The driver writes the IK, retiming and neighborhood reports to `OUTPUT`. Scene initialization and fresh-route generation are not included in the reported warm refinement time.
