# Tour 16: generated bead block and collision-witness retry

**On the same modeled clock, the three-stage generated block's time-aware jerk fell substantially after one collision-witness refinement, but the retry still fails the zero-penetration sampled check.** The block covers `weld:277–279`; its outer joint position, velocity and acceleration are anchored to the generated cubic. The preceding generated lead is unchanged. The [machine-readable comparison](tour16-bead-contact-refinement.json) records all seven axes, process checks and source locations.

| Measure | Generated block | Collision-witness retry | Evidence |
|---|---:|---:|---|
| Modeled duration | 52.458 s | 52.458 s | `generated_dynamics`, `retry_dynamics` |
| Track speed-normalized integrated squared jerk | 3.675 s⁻³ | 0.0281 s⁻³ | `integrated_speed_normalized_jerk_squared`, axis 0 |
| Sum for six arm axes | 0.4038 s⁻³ | 0.0244 s⁻³ | Same field, axes 1–6 |
| Peak track velocity / acceleration / jerk | 0.403 m/s / 0.614 m/s² / 15.20 m/s³ | 0.407 m/s / 0.240 m/s² / 0.279 m/s³ | Per-axis derivative peaks |
| Largest arm-axis velocity / acceleration / jerk | 0.150 rad/s / 0.229 rad/s² / 5.96 rad/s³ | 0.143 rad/s / 0.138 rad/s² / 0.813 rad/s³ | Per-axis derivative peaks; maxima may be on different axes |
| `weld:278` maximum timed-target error | 4.517 mm | 2.717 mm | `generated_process`, `retry_process` |
| `weld:278` modeled feed range | 0.659–13.313 mm/s | 2.219–8.271 mm/s | Same fields; nominal source feed is 5.00 mm/s |
| Sampled contacts | No sampled penetrating contacts | Three penetrating samples, worst −0.00209 mm | `generated_process`, `retry_contact_summary` |

These derivative peaks use numerical polynomial velocity extrema and cubic-segment acceleration/jerk; integrated squared jerk is integrated over each cubic piece. The speed normalization uses cell-declared track and arm velocity scales, not physical acceleration or jerk limits. The clock is modeled, and the resulting values are not controller measurements. The retry reports solver convergence but fails the independent zero-penetration check. Its remaining contact is in the prior `nozzle_link` / `part_028` episode; physical significance is not established.

The source is a previously generated motion, not a recorded intermediate joint seed. The retry adds only three collision times from the independently observed episode and retains the source clock and process constraints. Its generated lead still reverses about 0.057 mm along the sampled local target tangent. Neither finite samples nor this local block establish continuous clearance, complete-tour continuity, acceptable manufacturing deviation, controller interpolation or execution.

The comparison is reproducible with `weld_block_compare.py`, passing explicit `--generated`, `--retry`, `--audit`, `--joint-speeds` and `--output` paths. The precise local inputs are recorded in the JSON; the large candidate/scene artifacts remain local research inputs and are not part of this compact report. `test_weld_block_dynamics.py` checks derivative peaks and the exact integral against a known cubic.
