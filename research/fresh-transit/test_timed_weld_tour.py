"""Generated bead conversion preserves its clock, state and fixture."""

import numpy as np
import pytest
from timed_weld_tour import TimedWeldTourError, generated_bead_input


def test_generated_bead_keeps_nine_axis_endpoints_and_clock():
    states = np.arange(36, dtype=float).reshape(4, 9)
    states[:, 7:] = [0.7, 1.1]
    bead = generated_bead_input({"stage": "weld:5", "states": states.tolist()}, 2.4)
    np.testing.assert_allclose(bead["times_s"], [0.0, 0.8, 1.6, 2.4])
    np.testing.assert_array_equal(bead["joint_states_7"], states[:, :7])
    np.testing.assert_array_equal(bead["fixture"], [0.7, 1.1])
    assert bead["stages"] == [{"stage": "weld:5", "kind": "weld"}]


def test_generated_bead_rejects_moving_fixture_or_invalid_clock():
    states = np.zeros((4, 9))
    states[-1, 7] = 0.2
    with pytest.raises(TimedWeldTourError):
        generated_bead_input({"stage": "weld:5", "states": states.tolist()}, 1.0)
    states[-1, 7] = 0.0
    with pytest.raises(TimedWeldTourError):
        generated_bead_input({"stage": "weld:5", "states": states.tolist()}, 0.0)


def test_scene_restored_after_success_and_failure():
    from timed_weld_tour import preserve_scene

    class State:
        def __init__(self, joints):
            self.joints = joints.copy()

    class Scene:
        def __init__(self):
            self.joints = {"positioner": 0.5, "tailstock": 1.0}

        def getState(self):
            return State(self.joints)

        def getRevision(self):
            return 4

        def setState(self, joints):
            self.joints = joints.copy()

    scene = Scene()
    original = scene.joints.copy()
    with preserve_scene(scene):
        scene.setState({"positioner": 3.14, "tailstock": 0.8})
    assert scene.joints == original
    with pytest.raises(RuntimeError, match="solver failed"):
        with preserve_scene(scene):
            scene.setState({"positioner": 3.14, "tailstock": 0.8})
            raise RuntimeError("solver failed")
    assert scene.joints == original
