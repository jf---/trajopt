# Tour 16: active track limit at a generated lead

After applying one 1.5× modeled fixture-clock policy, the generated Tour 16
representation contains all 1,548 stages. Seventy-one of 72 fixture-adjacent
transits passed independent actual-curve sampled collision, position-extremum,
sourced-speed and fixture-join checks. The remaining transit, `transit:280`,
was rejected. Unchanged stages retain their earlier generated joint-linear
interpolation; process boundaries, interior corners and physical controller
states remain unresolved. This is not an accepted smooth tour.

The rejected transit began at the 0 m track bound with a modeled incoming track
velocity of −0.400 m/s from generated lead `weld:279`. Its converged spline
reached −4.8175 mm at 0.01867 s, before the first existing position collocation
at 0.03319 s. Six bounded actual-extremum witness rounds did not solve this
incompatible two-sided boundary; later iterates also overshot the upper track
bound. The complete [study evidence](tour16-track-boundary-study.json) keeps
these failures and modeled solver outcomes.

The active-limit rule sets only axes at a position bound to zero shared
velocity; it does not stop the entire robot. An initial generated joint-space
lead fit matched q/qd/qdd through the following transit and fixture and passed
finite collision and numerical position/speed checks, but departed the lead's
endpoint line by 12.49 mm and reversed 2.60 mm along its projection. That
exploratory lead is not accepted.

A subsequent generated Cartesian/track experiment interpolated the lead
endpoint poses in the workpiece frame, used a smooth generated track schedule
and solved collision-screened arm IK along it. The resulting actual lead's
maximum line departure was **0.434 mm**, with **0.033 mm** maximum backward
projection. At the shared bound the track velocity is zero while the arm still
moves. The lead and transit passed 2,050 finite collision samples and numerical
position and sourced-speed extrema; their modeled q/qd/qdd join and the
transit/fixture join matched. Generated lead TCP speed ranges **2.49–85.06
mm/s** on this modeled clock. The lead's feed variation, path tolerance,
controller interpretation, physical boundary states and fixture dynamics are
not approved or observed. The [follow-up assembly](tour16-fixture-transit-assembled.json) incorporates
the checked lead and transit without recomputing the other 71 transits. All 72
fixture-adjacent transits have finite collision, position and sourced-speed
checks and modeled fixture-join diagnostics. The 69 fixture stages were checked
separately under the same clock; the replacement lead was rechecked during
assembly. The other 1,406 stages—241 transits, 892 probe stages and 273
weld/lead stages—remain the previously sampled generated joint-linear path and
were not rechecked at 1,025 samples here. The assembled path reports 1,369
unresolved modeled velocity joins and 71 acceleration joins under a numerical
parity test. These counts do not establish physical significance; complete
process continuity and physical/manufacturing acceptance remain open.
