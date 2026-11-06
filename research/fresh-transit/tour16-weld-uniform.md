# Tour 16: uniform exploratory weld curves

One retained scene generated all 53 weld groups (274 lead and active-bead
stages) from the fresh candidate. Every group used the same policy: matched
generated outer endpoints, generated rail reference, source active-bead
orientation targets and at most three actual-error witnesses per group. The
rail reference is generated, not an optimized rail trajectory. Intermediate
recorded robot joints were not used as a seed.

All 53 groups were preserved. Fifty-two passed the configured sampled checks;
one failed the ordered-progress check at `weld:278`, where the actual curve's
source-chord projection moves backward by 0.050 mm. The target blend progresses
forward, and a separate 3D target-progress audit is pending. Its active-bead
modeled feed ranges from 0.659 to 13.313 mm/s against a source-derived nominal
5.000 mm/s; this variation needs correction or a process decision. Its maximum
time-aligned target-position error is 4.517 mm. Across all groups, the largest
sampled target-orientation error is 0.286° at `weld:203`. The maxima are
measurements, not established manufacturing tolerances.

The batch used 159 bounded witnesses in total. Curves were checked at 1,025
actual samples per stage, including contacts and joint positions; position
extrema were numerically evaluated, and speed checks were sampled. No sampled
penetration was reported. These are isolated group checks, not continuous
clearance or a complete tour: adjacent transits have not yet been reconnected,
and actual controller timing and process acceptance remain unknown.

[Compact evidence](tour16-weld-uniform-evidence.json) preserves all generated
curves, statuses, source-waypoint errors and measured process metrics. The
unabridged local research output is
`.superpowers/process-horizon16/weld-uniform-generated-orientation-all/results.json`.
The same explicit-input runtime described in [the research README](README.md)
ran `weld_local_batch_trial.py` with `--lead-policy matched-endpoints`,
`--witness-refinement`, `--orientation-witness-refinement`,
`--source-bead-orientation`, `--generated-rail-reference` and
`--verified-recovery`; scene and library paths must be supplied explicitly.
