# Tour 16: coalescing near blend transitions

**A shared generated-sample coalescing rule removed the short interval that prevented one local PIQP solve, while leaving important process-quality questions open.** It was tested on three generated bead neighborhoods with corrected fixture-before-kinematic-group ordering, one profile and unchanged modeled clocks. The [compact report](tour16-coalesced-bead-blocks.json) includes source and candidate stage checks, derivatives, constraint residuals and input locations.

| Generated source segment | Removed routine samples | Solver / sampled checks | Normalized integrated squared jerk, source → candidate |
|---|---:|---|---:|
| 1735 | 1 | Converged; sampled contact, position and sourced-speed checks pass | 0.00735 → 0.000648 s⁻³ |
| 1777 | 0 | Converged; no sampled contacts, but track position exceeds the configured feasibility band | 0.1845 → 0.0369 s⁻³ |
| 1835 | 1 | Converged; sampled contact, position and sourced-speed checks pass | 36.369 → 0.0506 s⁻³ |

The rule removes an unprotected routine generated sample only next to a protected blend transition, when their interval is below 0.001 of that stage's median sample spacing and the generated joint displacement is within the sourced axis-speed travel for that interval. It keeps stage boundaries, blend transitions and prior process witnesses. This is an exploratory representation rule, not a manufacturing tolerance. No recorded intermediate joint trajectory is used as the reference.

For 1835, the removed ordinary IK sample was 20 μs before a retained blend boundary. The reconstructed source preview had a maximum sampled workpiece TCP change of 1.16 × 10⁻⁹ m and orientation change of 2.55 × 10⁻¹⁰ rad, with no sampled penetrating contacts. This is a measured near-match, not exact preservation. The solver's first QP maximum Hessian entry fell from 7.59 × 10¹⁴ to 8.14 × 10⁷; the prior representation exhausted PIQP's iteration limit, while this trial converged. The solve overlapped another Tour 16 diagnostic, so its 11.24 s is not a controlled performance benchmark.

The original source process remains the relevant comparison. In 1835, weld `weld:315` timed-target error rose from 0.553 to 1.608 mm, modeled feed ranged from 3.47 to 11.14 mm/s against a 5 mm/s nominal, and sampled local target-tangent reversal was 0.039 mm. Its generated lead's target error rose from 0.010 to 3.217 mm and local reversal was 0.051 mm. In 1735, source-to-candidate target errors rose from 0.053/0.097/0.009 to 0.259/0.334/0.177 mm across the block, with feed approximately 4.79–5.28 mm/s. In 1777, the generated lead reversed 0.132 mm along its sampled target and the weld's raw track excess was 0.1175 mm versus the experiment's 0.100 mm feasibility band. A separate sampled contact scan of that weld passed; it applies because the checked cubic coefficients and fixture match the current candidate.

The target and collision checks use 1,025 samples per stage. Numerical polynomial extrema and cubic-piece jerk integrals characterize the modeled curves; they do not establish continuous collision clearance, physical machine dynamics, manufacturing tolerance, controller interpolation or complete-tour continuity. The source-generated curves remain preserved as separate candidates.

`weld_coalesced_summary.py` creates the compact report from explicit batch, preview, previous-check and contact-audit paths. Large generated inputs and full traces are local research artifacts listed in the JSON. `test_generated_knot_coalescing.py` checks that protected transitions survive and that a fast distinct sample is not removed.
