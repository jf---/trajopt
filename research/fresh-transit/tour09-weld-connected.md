# Tour 09: connected exploratory process curves

This research result assembles the nine generated probe groups, all three
exploratory weld clusters, the fixture stages and 14 transits into the fresh
Tour 09 candidate. It refines each of the six weld-adjacent transits once using
its actual neighboring q, qd and qdd. The other eight transit curves are
retained. No recorded intermediate robot or track joints seed the generated
motion. The detailed candidate and verifier artifacts remain in the local
`.superpowers/process-horizon09/weld-multi-connected-tour09/` and
`weld-multi-connected-tour09-verify/` directories. The
[compact verifier evidence](tour09-weld-connected-verified.json) preserves
per-stage checks and the six solver results.

The independent verifier evaluated 1,025 states on each of 67 actual curves:
68,675 samples, none with detected penetration. Numeric position extrema and
sampled sourced joint-speed checks passed; the largest sampled speed ratio was
0.8334. The 66 modeled stage joins matched in q, qd and qdd; the maximum
absolute residuals were 6.64e-15, 8.97e-14 and 5.40e-12 in the respective
axis units. There were no remaining joint-linear process interpolation corners
or source-bead order failures. These checks are finite-sampled collision
checks, not a continuous-collision proof.

The original bead endpoints and intermediate source anchors are retained. The
actual bead curves depart the source straight segments by at most **0.0390 mm**
and **0.00944°**, while modeled bead feed ranges **2.90–5.84 mm/s** against
roughly 5 mm/s nominal. The generated lead curves depart their former straight
references by up to **0.980 mm**; lead feed is not bead feed. Modeled search
cruise ranges **9.989–10.011 mm/s**. Probe hits remain hypothetical at the
commanded limit. No manufacturing tolerance or approval for the blended beads,
lead geometry, feed variation, or controller hit/correction behavior has been
supplied.

The clock is modeled and totals **153.27 s**. Fixture speed and acceleration
limits and physical outer boundary derivatives are unsupplied. The maximum
sampled track jerk is **48.0 m/s³** (transit 19); the maximum arm-axis jerk is
**170.8 rad/s³** (transit 0). Cubic piece boundaries need not have continuous
jerk. Thus C2 joins do not establish smoothness optimality or actual robot
execution. Scene and solver timings in the evidence were subject to concurrent
work and are not controlled performance measurements.

The assembly and checker are explicit opt-in research drivers:
`weld_multi_connect_trial.py` requires the verified fresh candidate, generated
probe batch, saved local-blend evidence, station and native libraries; pass its
result to `probe_cruise_multi_verify.py --exploratory-welds`. Both require
explicit input and output paths, and neither silently supplies physical
boundary derivatives. The environment and native prerequisites are in
[README.md](README.md). This result does not establish performance or process
acceptance for the other tours.
