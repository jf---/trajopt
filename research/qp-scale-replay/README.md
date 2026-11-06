# Matched PIQP cost-scaling replay

This opt-in diagnostic replays the same live SQP QP in two fresh native PIQP solvers. It changes only `preconditioner_scale_cost`, leaving the default 10 preconditioner iterations, tolerances, QP data and trust box unchanged. Original-space bound and trust-box violations and objective calculations are reported. The original SQP callback is stopped after the first matched replay; no returned trajectory is accepted.

For the generated Tour09 cluster 1202 first QP, both variants reported `solved`. Default cost scaling used 18 iterations, with maximum original-space bound violation 1.16e-8 and PIQP dual residual 2.44. Enabling cost scaling used 21 iterations, with violation 4.37e-11 and dual residual 4.16. Both had zero trust-box excess. [The evidence](tour09-first-qp-evidence.json) is a single-QP diagnostic and does not establish full-SQP convergence or motion quality.

The absolute QP objective suffers cancellation: the absolute terms total ~2.56e15, giving an epsilon-times-term-sum scale of ~0.57, larger than the difference of the absolute objective reports. The separately accumulated centered objective difference is −0.01033 (scaled minus default), with epsilon-times-term-sum ~3.89e-5. This is a rounding-scale indicator, not a rigorous bound. It does not justify selecting the scaled setting without a complete trajectory experiment.

Build with the research Pixi environment, explicit corrected SQP include/library, and PIQP package prefix. `SQP_INCLUDE`, `SQP_LIBRARY`, `PIQP_PREFIX`, `DEPENDENCY_PREFIXES`, `DEPENDENCY_LIB`, `MANIFEST`, `PYTHON`, and `OUTPUT_BUILD` must name the actual dependencies:

```bash
pixi run --as-is --manifest-path "$MANIFEST" cmake -S research/qp-scale-replay -B "$OUTPUT_BUILD" \
  -DCMAKE_PREFIX_PATH="$DEPENDENCY_PREFIXES;$PIQP_PREFIX" -DPython_EXECUTABLE="$PYTHON" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L$DEPENDENCY_LIB" \
  -Dpiqp_DIR="$PIQP_PREFIX/lib/cmake/piqp" -DSQP_SOURCE_INCLUDE="$SQP_INCLUDE" \
  -DSQP_LIBRARY="$SQP_LIBRARY"
pixi run --as-is --manifest-path "$MANIFEST" cmake --build "$OUTPUT_BUILD" -j 2
```

The consuming environment must also provide nanobind, Eigen and the TrajOpt/Ifopt dependency prefixes; use its normal linker search path. Add the resulting module and the existing research dependencies to `PYTHONPATH`; run `replay_tour.py --help` for the explicit station, fresh candidate, connected batch, original weld evidence and runtime-library arguments. The QP and two solver instances stay in native C++; the callback only requests the comparison and records its results.
