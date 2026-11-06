# SearchL cruise with coordinated nullspace motion

This opt-in trial uses the already verified fresh Tour20 probe0. It preserves
the four original Cartesian probe strokes and their ordered endpoints, and
generates new arm and track states with OPW IK rather than using recorded robot
states. At each Cartesian corner the torch slows to zero while a generated
shared q/qd/qdd allows arm and track nullspace motion. The two adjacent generated
transits are refined against the actual process endpoint derivatives.

The modeled search and retract clocks use smooth acceleration and deceleration
ramps around a cruise at the original modeled feed. The common policy uses a
ramp duration and a cruise duration each equal to half the source phase
duration; each search/retract phase therefore takes 1.5 times its source
modeled duration. This is a feasibility hypothesis, not a physical controller
clock or a claim that a physical search hits the commanded limit.

The [connected neighborhood evidence](probe-cruise-tour20-evidence.json)
covers six actual curves, each with 1,025 sampled collision checks. No
penetrating contact was detected; numerical position extrema and sampled
sourced joint-speed checks passed. All five modeled q/qd/qdd joins match to
numerical precision. The three internal rail velocities are approximately
+0.599, −0.971 and −1.504 mm/s. Every interior generated IK target had at
least eight available arm branches; sampled branch steps also met sourced
joint-speed checks.

The SearchL modeled cruise measured 9.993–10.007 mm/s around the source
nominal 10.0005 mm/s. Search duration increased from 1.900 to 2.850 s and
retract from 0.380 to 0.570 s. Maximum sampled SearchL departure from its
source line was about 3.2 nm; maximum deviation from the modeled time-progress
target was 0.39 µm. Sampled peak jerk and squared-jerk integrals are preserved
per axis in the evidence; axis order is track m/s³ then six arm rad/s³, with
integrals in the corresponding squared units over time. These samples are not
continuous jerk extrema or controller measurements.

The IK interpolation, collision and process checks are finite-sampled. The
actual hit, controller correction and approved process tolerances are unknown.
This is one probe neighborhood; it does not establish full-tour or all-tour
acceptance. `probe_cruise_trial.py --help` lists the explicit station, fresh
candidate, native library and new output paths. Run with the research Pixi
runtime documented in [README.md](README.md). Its timing is correctness-only,
not a controlled benchmark.
