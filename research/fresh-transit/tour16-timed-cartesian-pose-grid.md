# Timed Cartesian target on three generated weld blocks

This opt-in test adds native position and orientation residuals at modeled process times. Targets are the generated workpiece-relative process geometry and the source bead orientation; they are not recorded intermediate joint states. Position is normalized by each stage's target chord length in metres, orientation by one radian. These are objective scales, not manufacturing tolerances. The existing sampled hard line, ordered-progress, source anchors, collision rows, clock and stage-join constraints remain in place. The coalescing rule and all other solver settings are shared across cases.

Each block has three actual piecewise-cubic stages; the table reports the middle, bead stage. Feed is sampled TCP speed, against a modeled nominal 5 mm/s. Jerk is the sum of the seven axes' integral of squared jerk divided by sourced axis speed squared, in s⁻³. Weight zero is the earlier corrected-fixture coalesced baseline; weights 1 and 10 were run in one retained scene. All solves reported convergence, which is not process acceptance.

| Source segment | Pose weight | Maximum timed bead-target error (mm) | Bead feed range (mm/s) | Integrated normalized jerk (s⁻³) | Actual checks |
| --- | ---: | ---: | ---: | ---: | --- |
| 1735 | 0 / 1 / 10 | 0.334 / 0.124 / 0.065 | 4.79–5.28 / 4.83–5.24 / 4.93–5.10 | 0.000648 / 0.001009 / 0.001076 | All sampled contacts and limits clear |
| 1777 | 0 / 1 / 10 | 0.516 / 0.434 / 0.311 | 1.98–9.80 / 2.05–9.18 / 2.51–8.03 | 0.036919 / 0.035969 / 0.039259 | Weight 1 clears; 0 and 10 exceed the configured track-position band |
| 1835 | 0 / 1 / 10 | 1.608 / 1.282 / 0.940 | 3.47–11.14 / 3.25–8.74 / 3.33–8.01 | 0.050603 / 0.053006 / 0.051640 | All sampled contacts and limits clear |

A separate 1,025-state scan of 1777's weight-10 bead found no penetrating contacts after the main checker stopped at the track limit. The raw track excursion was 0.1275 mm against the existing 0.1 mm numerical feasibility band; physical significance is not established. The earlier baseline bead was separately sampled for contacts. Full values, axis dynamics, modeled durations, per-stage deviations, progress and solver status are in [the compact evidence](tour16-timed-cartesian-pose-grid.json).

For source 1835, sampled bead backward progress was 0.0395 / 0.0374 / 0.0130 mm for weights 0 / 1 / 10. Generated leads are curved, so their ordered progress is evaluated on the 3D target path, not simply projected onto an endpoint chord. None of these block-level checks establishes whole-tour jerk, process tolerances, controller timing, probe behavior or physical execution. Collision checking is finite sampling rather than a continuous clearance proof. The trajectory's clock and active fixture state were identical across the comparisons. Separate timing diagnostics overlapped; solve times in the evidence are contended and not a speed benchmark.
