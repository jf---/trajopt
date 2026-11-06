# Tour 16: shorter generated probe timing

All 223 fresh generated probe groups were checked with the same modeled
approach/withdraw duration factor of 1.512 and the previous 0.5 SearchL ramp
fraction. The 1.512 factor is the maximum derivative of the one-sided quintic
progress polynomial. The existing optimized SearchL candidates were reused;
their source geometry and endpoint data were checked before insertion. No
recorded intermediate robot joints were used as a seed.

The first run reported 13 boundary-check failures. On replay, the intended
shared q/qd/qdd values were identical; independent spline evaluations differed
by at most 1.13e-11 native units/s² in acceleration. After canonicalizing the
already-verified SearchL endpoints and applying a spacing-scaled binary64
evaluation allowance, those 13 groups were rechecked. All 223 groups passed
1,025 finite collision/position samples per phase and numerical position and
sourced-speed extrema checks. The original failures and raw deltas remain in
the local evidence. No physical limits or collision criteria changed.

For representative probes 0, 111 and 222, the shorter one-sided factor
reduced modeled group durations from 7.170, 10.920 and 8.569 s to 6.444,
9.468 and 7.572 s. Their sampled maximum arm jerk remained 7.49, 6.68 and
7.32 rad/s³; normalized integrated jerk changed from 1.908, 1.874 and 1.842
to 1.920, 1.876 and 1.846. The SearchL nominal cruise remained approximately
10 mm/s. A further experiment with 0.25 SearchL ramps saved 0.570 s per group,
but increased normalized integrated jerk approximately eightfold and arm peak
jerk about fourfold; it is not used in the selected profile.

These are modeled clocks, not measured controller time. Hits at the search
limit are hypothetical, sampled contacts do not prove continuous clearance,
and process and physical boundary acceptance remain unresolved. The groups
still need shared-transit integration and complete actual-curve tour checks.
[Compact evidence](tour16-probe-factor1512-evidence.json) includes per-phase
results, raw join observations and the representative grid; full generated
nodes and the original/recheck outcomes remain in the explicitly listed local
untracked outputs. Four-worker elapsed time was 79.75 s under contention; it is
not a clean speed comparison.
