# Tour 16: generated moving-track weld neighborhood

This exploratory test uses the independently sampled, freshly generated Tour 16
recovery. Its active bead stage `weld:204` moves the track from 1.6 to 0.8 m.
The test evaluates the generated track reference on the source-derived modeled
clock and solves collision-screened arm IK along it; the same policy permits a
constant track. It does **not** freely optimize the track or use recorded
intermediate robot/track states. The 0.8 m change occurs over 27.027 modeled
seconds. Complete raw outputs are retained under the local
`.superpowers/process-horizon16/` directory; the
[compact evidence](tour16-moving-track-weld.json) retains per-stage checks.

| Hypothesis | Max bead position departure | Max orientation departure | Modeled bead feed |
| --- | ---: | ---: | ---: |
| Local orientation blends | 0.0856 mm | 2.643° | 4.989–5.019 mm/s |
| Source orientation targets | 0.0906 mm | 0.286° | 4.941–5.072 mm/s |

Both use local exploratory Cartesian blends through source anchors, 151 generated
IK nodes plus three witnesses, and 1,025 actual-curve collision samples per
stage. Source-orientation targets reduce departure but the fitted joint spline
still departs between nodes. On this example, an additional bounded policy that
ranks position and orientation errors relative to each one's observed maximum
selected the same three witnesses as position-only and produced the same
result. Its witness pass took 0.83 s rather than 0.62 s under possible CPU
contention. Those numbers are not clean performance benchmarks. No process
acceptance tolerance has been provided.

The source-orientation result was connected to both neighboring transits and
the following eased fixture chain. The independent verifier checked 14 actual
stages and 14,350 collision samples; no sampled penetration was detected.
Position extrema, sampled sourced robot speeds and ordered bead progress
passed; all 13 internal modeled q/qd/qdd joins matched. The two transit
optimizations converged. The preceding withdrawal's boundary velocity is a
modeled joint-linear secant and its acceleration is a free optimization
variable; neither is observed physical state. Fixture timing and speed and
acceleration limits remain unsourced. Finite collision samples do not prove
continuous clearance, and the fitted orientation, feed and generated leads
have not been approved for manufacturing or controller execution.

This does not cover all Tour 16. Its 53 weld clusters include two other
moving-track lead-outs. The current fixture easing rejects the separate
`fixture:40` chain because the full chain is not a single nine-axis chord;
its ordered target geometry needs a general continuous treatment. The
experimental drivers accept explicit station, recovery, native-library and
output paths; `weld_local_batch_trial.py` has explicit
`--verified-recovery --generated-rail-reference --source-bead-orientation`
options, and `weld_neighborhood_connect_trial.py` independently verifies the
connected candidate. See [README.md](README.md) for the research runtime.
