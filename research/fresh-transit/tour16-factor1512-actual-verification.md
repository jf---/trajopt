# Tour 16: shorter-clock actual-curve verification

The independently sampled check covered all 1,548 stages of the shorter-clock
assembly: 1,586,700 actual states across four separately owned scenes. No
sampled collision, joint-position or sourced-speed failures were detected.
The four-process wall time was 115.52 s under contention; it is not a clean
performance comparison or continuous-clearance guarantee. The 223 probe groups
also have separate actual Cartesian checks in
[the probe study](tour16-probe-factor1512.md).

The assembly retains 273 generated joint-linear weld/lead stages. At 4,095
sampled internal corners their velocities are undefined under that
interpolation. The largest modeled stage-boundary velocity gaps remain
0.400 m/s on the track (`weld:315` → `weld:316`) and 0.351 rad/s on an arm
axis (`weld:53` → `weld:54`). Thus sampled collision and speed checks do not
establish a smooth tour. The 53 independently checked exploratory weld groups
in [the separate weld study](tour16-weld-uniform.md) have not been integrated
into this verification.

`transit:160` initially contacted `nozzle_link` and `part_025` during the move
from probe withdrawal to generated lead. The zero-margin checker found 20
contact samples, worst signed distance −0.186 mm. A bounded retry kept the
original 33 collision times and added the sampled contact episode's entry,
deepest and exit times. It preserved the modeled shared boundaries and passed
the independent 1,025-state transit check. The original failure, contact pair,
raw distances and retry outcome remain in the [evidence](tour16-factor1512-actual-verification.json).
No allowed-contact rule or collision margin was changed.

Modeled tour duration is 5,160.803 s, versus 5,449.913 s in the previous
probe-clock assembly. The source-generated modeled clock is 4,479.915 s.
These are not measured robot cycle times. The first and final transit outer
velocities and accelerations are explicit optimization variables with an
exploratory soft speed preference; the physical cell values remain unknown.
Search hits are hypothetical, and process geometry and controller acceptance
remain unresolved.
