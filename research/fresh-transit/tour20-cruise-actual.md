# Tour20: all probes with modeled search cruise

The same ramp/cruise/nullspace policy from [probe0](probe-cruise-tour20.md) was
applied to all three previously generated probes in one scene. The four shared
transits were each refined once against both neighboring actual q/qd/qdd; the
last connects to the previously regenerated tangent lead. The bead spline and
its modeled clock remain unchanged. The [batch evidence](tour20-cruise-batch-evidence.json)
contains the generated IK samples, boundary derivatives, process checks and
four transit curves. It uses generated robot states only, not recorded robot
states as seeds.

The independent [full-tour actual-curve check](tour20-cruise-actual-evidence.json)
reconstructed and sampled all 20 stages, 1,025 samples per stage. It found no
penetrating contacts; numerical position extrema and sampled sourced joint
speeds passed, with maximum sampled speed ratio 0.828. The 19 modeled stage
joins matched q/qd/qdd to numerical precision and no joint-linear process
corners remain. Collision checking remains finite-sampled. Search line/order,
original generated endpoints and the unchanged bead were checked; no physical
process tolerance was invented.

The three SearchL modeled cruise ranges were approximately 9.990–10.008 mm/s
around their ~10 mm/s nominal feed. The clocks ramp to and from zero TCP speed
at each exact Cartesian corner, with arm and track allowed to continue in the
modeled nullspace. Each search and retract takes 1.5 times its original modeled
duration. The resulting tour clock is 44.02 s, compared with 46.59 s for the
earlier fixed-joint-stop diagnostic; neither is measured robot cycle time. The
unchanged bead remains at ~5 mm/s without an in-weld stop. The generated tangent
leads still have ~5–79 mm/s modeled feed variation and unknown process
admissibility. Hypothetical hits at commanded limits, controller correction,
outer physical derivatives and manufacturing acceptance remain unresolved.

Sampled whole-tour peak jerk is 5.73 m/s³ on the track at transit7 and
34.57 rad/s³ on an arm joint at lead-in. These are modeled sampled values, not
hard bounds or controller data. The unchanged lead policy is a separate source
of peak jerk; no all-24 quality or controller-execution claim is made.

`probe_cruise_batch_trial.py` performs the all-probe build in one prepared
scene. `actual_tour_trial.py --probe-cruise-batch` independently reconstructs
and checks the actual curves. Their `--help` lists explicit fresh candidate,
lead and weld evidence, station, tour-index, native libraries and output
inputs; run in the research Pixi environment described in [README.md](README.md).
The clock's exact endpoint correction is a floating-point boundary fix, not a
relaxation of interior process checks. Timing measurements are correctness-only
and may be contended.
