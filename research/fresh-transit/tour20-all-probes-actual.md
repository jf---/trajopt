# Tour20 actual-curve assembly

This research trial assembles a previously fresh, independently verified Tour20
candidate with all three generated probes coordinated by the same fixed-path
diagnostic policy and a regenerated tangent-lead weld neighborhood. Four
probe-adjacent transits are refined once in a retained scene using the actual
q/qd/qdd of both neighbors; transit3 connects to the regenerated lead-in. The
other weld-adjacent transit comes from the separately checked connected-weld
experiment. The original generated bead spline and modeled clock are retained.

The [probe-batch result](tour20-probe-batch-evidence.json) preserves the four
refined transit curves and solver statuses. The independent
[actual-curve report](tour20-all-probes-actual-evidence.json)
records all 20 stages and 20,500 sampled states. No penetrating contact was
detected; numerical spline position extrema, linear extrema, and sampled
sourced arm/track speed checks passed. Maximum sampled axis speed was 0.828
of its sourced limit. All 19 modeled q/qd/qdd stage joins matched to numerical
precision, and the assembly has no remaining joint-linear interpolation
corners. Each process stage was sampled against the line between its endpoints;
original probe and bead endpoints were also checked against the generated
source. Collision remains finite sampling, not continuous-clearance proof.

The modeled duration is 46.59 s and is not controller time. There are nine
explicit modeled internal probe stops; only the SearchL reversal is directly
explained by reversing the search direction. The other stops belong to this
fixed-joint-path ablation and are not established as physically necessary.
SearchL feed varies from zero to roughly 10 mm/s, rather than the previous
uniform modeled feed. The previously verified input assumes hypothetical hits
at commanded limits; actual hit positions and controller correction are not
supplied. Generated lead admissibility and process tolerances are unknown; the
leads in this experiment vary from about 5 to 79 mm/s. The bead stays at about
5 mm/s without an in-weld stop. Physical outer derivatives are unknown, and no
whole-tour jerk-continuity or manufacturing-acceptance claim is made.

`probe_batch_trial.py` creates the all-probe shared-transit result from the
same fresh candidate and connected tangent-lead evidence. `actual_tour_trial.py`
then reconstructs all actual curves and rechecks the complete tour. Both
drivers require explicit station, tour index, candidate, evidence, native
library and new output paths; `--help` lists the arguments. Run in the research
Pixi runtime described in [README.md](README.md), with the native spline and
collision bindings on the Python path. Saved inputs and native libraries are
development dependencies, not an installed product. The correctness run
overlapped other work, so its elapsed times are not performance measurements.
