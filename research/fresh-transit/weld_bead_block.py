"""Select one unresolved generated bead and its immediate process neighbors."""


class WeldBeadBlockError(RuntimeError):
    """The requested generated bead cannot form a bounded neighborhood."""


def bead_block_indices(stages, unresolved):
    """Return the half-open three-stage block around the sole unresolved bead."""
    if len(unresolved) != 1:
        raise WeldBeadBlockError("exactly one unresolved bead is required")
    matches = [i for i, stage in enumerate(stages) if stage["stage"] == unresolved[0]]
    if len(matches) != 1 or stages[matches[0]]["kind"] != "source_bead":
        raise WeldBeadBlockError("unresolved stage is not a unique source bead")
    index = matches[0]
    if index == 0 or index + 1 >= len(stages):
        raise WeldBeadBlockError("both neighboring process stages are required")
    return index - 1, index + 2


def representative_bead_block_indices(stages):
    """Use the middle source bead under one shared selection policy."""
    beads = [stage["stage"] for stage in stages if stage["kind"] == "source_bead"]
    if not beads:
        raise WeldBeadBlockError("a source bead is required")
    return bead_block_indices(stages, [beads[len(beads) // 2]])
