"""A verified fresh tour can opt into the common transit refiner."""

import fresh_tour_pipeline as pipeline
import numpy as np


def test_verified_tour_calls_explicit_refiner(monkeypatch):
    names = tuple(f"j{i}" for i in range(9))
    q0 = np.zeros(9)
    q1 = q0.copy()
    q1[0] = 0.1
    segments = [{"stage": "transit:0", "states": [q0.tolist(), q1.tolist()]}]
    clock = {"stages": [{"stage": "transit:0", "duration_s": 1.0}]}
    monkeypatch.setattr(pipeline, "prepare_fresh", lambda *args: (segments, clock, np.ones(7)))
    knots = np.linspace(0, 1, 7)
    nodes = np.zeros((7, 7))
    nodes[:, 0] = knots * 0.1
    velocity = np.zeros(7)
    velocity[0] = 0.1
    calls = []

    def chosen(*args, **kwargs):
        calls.append(args[3])
        return [
            {
                "stage": "transit:0",
                "knots_s": knots.tolist(),
                "nodes": nodes.tolist(),
                "duration_s": 1.0,
                "v0": velocity.tolist(),
                "v1": velocity.tolist(),
                "a0": [0.0] * 7,
                "a1": [0.0] * 7,
                "fixture": [0.0, 0.0],
            }
        ]

    class Checker:
        def contacts(self, _q):
            return []

    monkeypatch.setattr(pipeline.StateChecker, "build", lambda *args: Checker())

    class Limits:
        joint_limits = np.tile([-10.0, 10.0], (9, 1))

    class Group:
        def getLimits(self):
            return Limits()

    class Env:
        def getJointGroup(self, *args):
            return Group()

    scene = {"inputs": (Env(), None, None, None, names, None, None, None, None, None, None, None)}
    result = pipeline.refine_checked_tour(
        scene,
        20,
        {},
        {"status": "independently_sampled_tour"},
        {},
        {},
        boundary_policy=lambda *_: (np.zeros(9), np.zeros(9)),
        transit_solver=chosen,
    )
    assert calls == [segments]
    assert result["transits"][0]["actual_curve_check"]["sample_count"] == 1025


def test_explicit_fixture_clock_policy_receives_verified_stages(monkeypatch):
    import pytest

    segments = [{"stage": "fixture:0", "states": [[0] * 9, [0] * 9]}]
    clock = {"stages": [{"stage": "fixture:0", "duration_s": 1.0}]}
    monkeypatch.setattr(pipeline, "prepare_fresh", lambda *args: (segments, clock, np.ones(7)))

    class SeenClock(Exception):
        pass

    def policy(given_stages, given_clock):
        assert given_stages is segments and given_clock is clock
        raise SeenClock

    scene = {
        "inputs": (
            None,
            None,
            None,
            None,
            tuple(f"j{i}" for i in range(9)),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }
    with pytest.raises(SeenClock):
        pipeline.refine_checked_tour(
            scene,
            9,
            {},
            {"status": "independently_sampled_tour"},
            {},
            {},
            boundary_policy=lambda *_: None,
            clock_policy=policy,
        )
