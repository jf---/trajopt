"""A fixture-specific kinematic group must be built after setting the fixture."""

from fixture_group_order import fixture_joint_group


def test_group_uses_requested_fixture_before_construction():
    class Environment:
        def __init__(self):
            self.state = {}

        def setState(self, state):
            self.state = dict(state)

        def getJointGroup(self, name):
            return name, self.state.copy()

    env = Environment()
    assert fixture_joint_group(env, "arm", ("positioner", "tailstock"), (3.0, 1.1)) == (
        "arm",
        {"positioner": 3.0, "tailstock": 1.1},
    )
