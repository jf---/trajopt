# Tour09: multi-cluster probe cruise experiment

This trial starts from a previously fresh, independently verified candidate.
It applies the same generated SearchL ramp/cruise and arm/track nullspace
policy to all nine probes in one scene, retains the five ordered fixture stages
and original generated welds, and refines each of the 14 shared transits once.
Original robot joint rows are not used as planning seeds. Probe curves are
marked generated/unverified until the separate actual-curve verifier runs.

The [generated result](tour09-cruise-generated.json) reports all nine groups
and 14 converged transit SQPs. The independent [verification](tour09-cruise-verified.json)
evaluated 67 stages and 68,675 states. No penetrating contact was detected;
numerical position-extrema and sampled sourced arm/track speed checks passed,
and the eased fixture stages and their modeled joins were retained. Across the
nine searches the sampled modeled cruise ranged from 9.989 to 10.011 mm/s;
sampled probe paths preserved their order and were checked against the source
Cartesian lines and endpoints. Collision clearance remains finite-sampled.

The overall modeled continuity result is **failed**, not accepted: the
unchanged joint-linear welds contain 180 modeled interpolation corners, and
nine weld-stage joins have velocity discontinuities, the largest about
0.042 rad/s. The fixture dynamic limits, controller timing, measured SearchL
hits and manufacturing acceptance remain unknown. The 153.27 s modeled tour
clock is not robot cycle time. Among the checked polynomial curves, sampled
peak track jerk was 48.0 m/s³ and peak arm jerk 170.8 rad/s³; jerk at the
retained weld corners is undefined in this model.

The [prior rejected attempt](tour09-cruise-prior-failure.json) stopped at
search:7 with the track 0.667 mm outside its configured position band. Its
unconstrained zero-TCP projection chose an outward rail velocity at the rail
limit. The revised global projection enforces the two-sided interior-join
velocity condition at active bounds, preserves the TCP equation and sourced
speed limits, and enforces inward acceleration curvature without inventing
acceleration ceilings. It retains compatible nullspace motion on other axes;
the resulting entire actual curve is still independently checked.

`probe_cruise_multi_trial.py` produces the unverified groups and shared
transits; `probe_cruise_multi_verify.py` performs the one explicit full
collision/position/process pass. The drivers require explicit fresh candidate,
station, tour-index, native-library and new output paths; `--help` lists them.
Use the research Pixi runtime described in [README.md](README.md). The result
is a modeled, contended correctness trial, not a performance benchmark or an
all-tour result.
