# Tour 16: unknown outer derivatives

The two outer transits have supplied positions but no measured outer velocity or
acceleration. This experiment gives those derivatives optimization variables,
keeps each fixture fixed, and matches q, qd and qdd to the neighboring modeled
process curve. Zero is only the initial optimizer value; it is not a stop
constraint or a statement about the physical cell. The terminal neighbor is
still joint-linear, so its modeled derivative is provisional.

The native nonuniform quintic model carries the free derivative variables into
its jerk, position, velocity and actual-spline collision Jacobians. Its C++
contracts compare the model against fixed-boundary curves and compare the
collision and cost Jacobians with finite differences. Each trial independently
checks 1,025 actual curve states, numerical position and speed extrema, and the
modeled process join. All cases below passed these checks without sampled
penetration; they are not continuous-clearance or controller-execution proof.

The outcome depends strongly on the initial SQP trust box. Without an outer
speed preference, enlarging the box from 0.1 to 1 to 4 changes the normalized
integrated jerk from 157.0 to 38.4 to 14.3 for the first transit and from
422.9 to 68.7 to 17.7 for the final one. At box 4 the optimizer drives an
outer axis to its cell-declared speed limit. The smaller-box solution is not a
credible optimality conclusion merely because the solver reports convergence.

An explicitly exploratory soft term uses
`beta * sum((outer_velocity / sourced_speed)^2)` with `beta = 100 s^-3`,
compared against the integrated `sum((jerk / sourced_speed)^2) dt`. At box 4,
first/final integrated jerk becomes 87.45/80.99, and the largest endpoint speed
ratios are 0.371/0.440. For this same beta objective, their totals are
117.3/116.6, compared with 315.6/193.3 for the no-preference box-4 curves.
This illustrates the tradeoff; it does not establish an acceptable start state
or select a production weight. An actual controller state is needed for that.

The [evidence](tour16-free-boundary-evidence.json) includes all five settings,
modeled curves, raw extrema, sampled collision results, velocity provenance and
local source paths. Timings were contended and are not a performance comparison.
The assembled tour still has 273 unrefined weld/lead stages and has not received
an independent full actual-curve verification. Modeled source and assembled
clocks are 4,479.915 and 5,449.913 s; neither is measured controller time.
