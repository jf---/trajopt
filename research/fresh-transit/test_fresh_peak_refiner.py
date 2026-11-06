"""The same explicit jerk profile covers every generated transit once."""

from fresh_peak_refiner import refine_all_peak_transits


def test_all_transits_use_one_policy_and_keep_neighboring_process_stages():
    stages = [
        {"stage": "transit:0", "states": [[0] * 9, [0] * 9]},
        {"stage": "search:0", "states": [[0] * 9, [0] * 9]},
        {"stage": "transit:1", "states": [[0] * 9, [0] * 9]},
    ]
    clock = {"stages": [{"stage": x["stage"], "duration_s": 1.0} for x in stages]}
    calls = []

    def solver(*args, **kwargs):
        selected = [s["stage"] for s in args[3] if s["stage"].startswith("transit:")]
        calls.append((selected, args[3][1]["stage"], kwargs))
        return [{"stage": selected[0]}]

    got = refine_all_peak_transits(
        None,
        None,
        (),
        stages,
        clock,
        (),
        solver=solver,
        start_velocity=[0] * 9,
        end_velocity=[0] * 9,
    )
    assert [x["stage"] for x in got] == ["transit:0", "transit:1"]
    assert [x[0] for x in calls] == [["transit:0"], ["transit:1"]]
    assert all(x[1] == "search:0" for x in calls)


def test_unknown_outer_acceleration_remains_free_and_interior_is_modeled():
    import numpy as np

    stages = [
        {"stage": "transit:0", "states": [[0] * 9, [0] * 9]},
        {"stage": "search:0", "states": [[0] * 9, [0] * 9]},
        {"stage": "transit:1", "states": [[0] * 9, [0] * 9]},
    ]
    clock = {"stages": [{"stage": x["stage"], "duration_s": 1.0} for x in stages]}
    boundaries = []

    def solver(*args, **kwargs):
        selected = [i for i, x in enumerate(args[3]) if x["stage"].startswith("transit:")][0]
        boundaries.append(kwargs["acceleration_resolver"](selected))
        return [{"stage": args[3][selected]["stage"]}]

    refine_all_peak_transits(
        None,
        None,
        (),
        stages,
        clock,
        (),
        solver=solver,
        start_velocity=[0] * 9,
        end_velocity=[0] * 9,
    )
    assert boundaries[0][0] is None
    np.testing.assert_array_equal(boundaries[0][1], np.zeros(9))
    np.testing.assert_array_equal(boundaries[1][0], np.zeros(9))
    assert boundaries[1][1] is None
