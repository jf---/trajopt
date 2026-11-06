# Modeled Cartesian velocity fidelity

The opt-in native velocity objective is prepared for the generated-weld experiment. It compares the workpiece-relative TCP displacement and generated target displacement between consecutive existing process witnesses. For interval Δt, the three-component residual is

```text
r = ((p_actual(t1) - p_target(t1)) - (p_actual(t0) - p_target(t0))) / (Δt * v_ref)
cost = λ * (Δt / T_ref) * ||r||²
```

The per-stage reference speed is the source-stage chord length divided by its modeled duration, in m/s, and the reference time is that duration. They normalize the soft objective; neither is a physical feed limit or manufacturing tolerance. Cartesian translation errors are rotated from each desired pose frame into the common workpiece frame before differencing. The result is average velocity error over each interval, not a guarantee about instantaneous feed between samples.

The implementation adds no trajectory knots and uses the same union of process witnesses and generated knot times as the existing process constraints. The prior timed-pose, time-aware joint-jerk, source-anchor/order and collision terms remain unchanged. The native implementation delegates to the existing analytic Cartesian pose residual and Jacobian; it has no Python callback during cost evaluation. A C++ test covers fixed and moving target frames, changing target orientations, nonuniform clock spacing, finite-difference Jacobians, invalid interval handling and an Ifopt squared-cost consumer. A Python binding test exercises the term in a real small Ifopt problem.

For the pending same-scene Tour09 comparison, use the existing `probe_cruise_multi_verify.py` with the explicit fresh generated candidate, checked connected batch and original generated weld evidence. The common exploratory settings are `--timed-pose-modes bead --timed-pose-weights 100 --timed-velocity-weights 0 1 10 --coalescing-fraction 0.05`. The three λ values run on every weld cluster in one prepared scene. Bead-only windows preserve each generated lead and its modeled q/qd/qdd boundary; weight zero retains the previous objective. The grid reports local actual-curve checks and does not select a candidate for whole-tour verification. No actual-cell scene test has been run for this cost yet.

Each velocity interval evaluates two native Cartesian pose terms in addition to the existing costs. Its runtime impact has not been measured. Independent sampled process path, feed, collision, extrema and modeled joins remain necessary, and physical robot/controller boundaries and process acceptance are unknown.
