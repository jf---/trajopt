# Original probe corners and nullspace motion

The original probe task consists of five workpiece-frame poses: stand-off,
search start, search limit, search start, stand-off. The source extractor
recovers these with FK; recorded robot rows are not used to seed new planning.
The selected phases are MoveL approach, SearchL, MoveL retract and MoveL
withdrawal. Search results are not measured here; hitting the commanded limit
is a hypothesis, and controller correction and blend semantics are unknown.

The [source tangent audit](probe-original-joins-20-09.json) evaluates both
position and orientation tangents at 12 original probes from Tours20 and09 in
one scene. For Tour20, the three approach-to-search position direction cosines
are approximately 0.652, −0.014 and 0.120; the reverse legs produce the
corresponding departure corners. All three search/retract direction pairs
reverse. For Tour09, the nine original probes also have noncollinear source
directions under exact interpolation; one pair has cosine 0.9998, whose
physical significance is unestablished. The classifier reports source
translation and rotation separately, using only a machine-scale comparison band;
it does not invent an allowed process deviation.

Under exact piecewise Cartesian geometry, continuous TCP velocity must reach
zero at a noncollinear join or a reversal. That does **not** imply all robot
joints must stop: arm and rail velocities may remain in the Jacobian nullspace.
A process-authorized blend could change the corner; no such blend tolerance has
been supplied. The classifier does not assign physical controller behavior.

The [generated Tour20 probe0 nullspace result](probe-nullspace-tour20.json)
projects a generated, seven-axis reference velocity onto a zero workpiece TCP
twist using sourced joint-speed weights and explicit task-unit scaling. The
independently finite-differenced FK Jacobian agreed after unit scaling. The
result includes −0.971 mm/s modeled rail motion with coordinated arm motion,
and sampled sourced-speed ratios remain below 0.00054. A straight-joint local
model over ±11.875 ms had no detected penetrating contacts in 1,025 samples,
no position excess, and maximum sampled TCP drift of 5.7e−11 m. This is a local
feasibility observation, not a full SearchL curve, a measured hit, an actual
controller boundary, or evidence that the required process feed is preserved.

`probe_source_join_trial.py` evaluates original joins for multiple `--tour`
arguments in one scene. `probe_nullspace_trial.py` evaluates one selected
`--probe` from an independently verified fresh candidate. Both require the
explicit station, tour index, corrected research libraries and new output
directory described in [README.md](README.md); run with `--help` for arguments.
The completed runs are correctness-only and may overlap other work.
