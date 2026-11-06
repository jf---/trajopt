# Tour09 source-bead corners and exploratory blends

The [source join audit](tour09-weld-source-joins.json) distinguishes original
bead geometry from generated leads. The five-pose bead has three internal
Cartesian turns of approximately 18.70°, 9.02° and 26.94°. Exact straight
segments cannot carry continuous nonzero TCP velocity through those corners;
arm/track redundancy does not change that task-space fact. A process-authorized
blend or zero TCP velocity is needed. No physical process tolerance or
controller blend behavior has been supplied.

An initial globally straight tangent-lead hypothesis was rejected: its first
lead had approximately 50.5 mm torch/workpiece penetration. The
[branch audit](tour09-weld-lead-branch-audit.json) found no clear generated IK
branch at 10 of 18 sampled nodes on that lead; all eight branches at the first
node had the same part_013/torch_link contact. For that fixed target pose and
fixture, changing the robot branch cannot remove the rigid torch contact. This
does not prove that other lead geometries or continuous connections are
infeasible. The rejected detailed artifacts remain in the local research data.

The separate [matched-endpoint local-blend experiment](tour09-weld-local-witness.json)
retains the previously generated outer lead endpoints and bends the leads toward
the source bead tangents. Within the five-pose bead it uses local Cartesian and
orientation blends through every original waypoint; the two straight beads
keep their original target lines. One uniform, bounded witness pass adds at most
three targeted IK nodes per bead. It reduced the largest actual-vs-blend target
departure from 0.657 to 0.082 mm for the five-pose bead, and from 0.0358 to
0.0095 mm and 0.0339 to 0.0090 mm for the straight beads. Counts changed from
102 to 105 nodes and 49 to 52 nodes per straight bead; the extra correctness
pass took about 0.59, 0.30 and 0.29 seconds under possible contention.

In the sampled results, all source waypoints are retained, modeled q/qd/qdd
joins match, bead progress is ordered, and 1,025 collision samples per stage
found no penetration. Numeric position extrema and sampled sourced joint speeds
pass. The five-pose bead departs its original lines by up to 0.039 mm and its
orientation by up to 0.0095°; its modeled bead feed ranges 2.90–5.84 mm/s.
The two straight beads have about 0.004 mm maximum sampled line departure.
Some curved leads move against the old straight-lead projection because the
new target itself turns; this is not treated as reversal along the new curve.

These are exploratory, finite-sampled weld neighborhoods. Feed, geometric
departure, collision clearance between samples, lead admissibility and
manufacturing acceptance are unresolved. The follow-up [complete Tour 09
assembly](tour09-weld-connected.md) connects the neighboring transits and
checks all actual curves. `weld_local_batch_trial.py` requires explicit station, fresh
candidate, native libraries, output and `--lead-policy matched-endpoints`; add
`--witness-refinement` for this bounded pass. The research Pixi runtime and
native binding paths are described in [README.md](README.md).
