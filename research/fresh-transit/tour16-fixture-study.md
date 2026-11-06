# Tour 16 fixture-axis motion: bounded experiments

The fresh Tour 16 candidate contains 69 fixture stages in 36 contiguous chains.
The existing single-chord timing model rejects the two noncollinear chains
beginning `fixture:40` and `fixture:58`. An unconstrained nine-axis spline can
interpolate their ordered targets, but it exceeded sourced track speeds; at
`fixture:620` it also reversed the prescribed positioner direction. It was
rejected. The separate native derivative-bound experiment enforces target knots,
sourced robot speeds and fixture-axis winding, without making the track or arm
monotone. Its 36-case duration/soft-jerk grid solved both noncollinear chains at
1.5× and longer modeled durations, but never solved `fixture:620` in that grid.
The [compact evidence](tour16-fixture-study.json) preserves all statuses and
residuals. Detailed outputs remain under local `.superpowers/process-horizon16/`.

An independent collision audit used a generated Tour 16 nozzle/part contact:
the nine-axis collision group includes both moving links, the native and
independent signed distances agreed, and analytic track, arm and positioner
Jacobians agreed with central differences. This checks one stable contact, not
continuous clearance.

A separate native fixture-axis quintic parameterization preserves each ordered
positioner and tailstock target. Its intermediate velocities are nonzero when
neighboring commands agree, and become zero at a reversal or hold. It uses zero
modeled acceleration at target boundaries; physical timing, fixture limits and
controller behavior remain unsupplied. Track and arm remain on the previously
generated nine-axis spline and are not forced monotone. This is a trajectory
hypothesis, not free track/arm optimization.

On the three selected chains, 1× duration passes the target, collision and
sourced robot-speed checks for `fixture:620`, while the other two exceed the
track-speed limit. One uniform 1.5× duration passes those same checks for all
three, using 2.60 s, 3.39 s and 6.20 s of modeled time respectively. The
interior path and clock are changed: departures from the earlier generated
chord timing reach 0.179 m track and 0.146 rad positioner. Sampled contacts,
numerical velocity extrema and target preservation do not establish controller,
fixture-dynamics or manufacturing acceptance. Neighboring transit continuity
and whole-tour verification remain outstanding.
