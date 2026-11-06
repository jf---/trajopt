from probe_transit_parallel import expected_transits


def test_shared_transit_is_selected_once_after_both_probe_groups():
    labels = [
        "transit:0",
        "approach:0",
        "search:0",
        "retract:0",
        "withdraw:0",
        "transit:1",
        "approach:1",
        "search:1",
        "retract:1",
        "withdraw:1",
        "transit:2",
    ]
    recovery = {"verified": {"expanded": {"segments": [{"stage": label} for label in labels]}}}
    groups = []
    for i in range(2):
        groups.append(
            {
                "status": "sampled_group_physical_and_process_acceptance_unresolved",
                "process": [
                    {"stage": f"{phase}:{i}"}
                    for phase in ("approach", "search", "retract", "withdraw")
                ],
            }
        )
    assert expected_transits(recovery, {"groups": groups}) == [
        "transit:0",
        "transit:1",
        "transit:2",
    ]
