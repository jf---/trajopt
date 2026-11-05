import numpy as np
import pytest
from native_transits import TransitContractError, refine_transits


def stages():
    first = np.zeros((2, 9))
    second = first.copy()
    second[0, 0] = 1.0
    return [
        {"stage": "approach:0", "states": first.tolist()},
        {"stage": "weld:0", "states": second.tolist()},
    ]


def test_rejects_broken_nine_axis_join_before_scene_use():
    with pytest.raises(TransitContractError, match="join mismatch"):
        refine_transits(
            None,
            None,
            list("abcdefghi"),
            stages(),
            {
                "stages": [
                    {"stage": "approach:0", "duration_s": 1.0},
                    {"stage": "weld:0", "duration_s": 1.0},
                ]
            },
            np.ones(7),
        )


def test_rejects_missing_timing_or_speed_before_scene_use():
    with pytest.raises(TransitContractError, match="positive modeled durations"):
        refine_transits(
            None,
            None,
            list("abcdefghi"),
            stages(),
            {
                "stages": [
                    {"stage": "approach:0", "duration_s": 0.0},
                    {"stage": "weld:0", "duration_s": 1.0},
                ]
            },
            np.ones(7),
        )
    with pytest.raises(TransitContractError, match="sourced speeds"):
        refine_transits(
            None,
            None,
            list("abcdefghi"),
            stages(),
            {
                "stages": [
                    {"stage": "approach:0", "duration_s": 1.0},
                    {"stage": "weld:0", "duration_s": 1.0},
                ]
            },
            np.zeros(7),
        )
