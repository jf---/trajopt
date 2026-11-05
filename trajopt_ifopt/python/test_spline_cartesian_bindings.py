"""Native Cartesian process constraints are usable in real Ifopt problems."""

import _quintic_spline as native
import numpy as np
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics.tesseract_common import GeneralResourceLocator, Isometry3d
from tesseract_robotics.tesseract_environment import Environment

URDF = """<robot name="line" xmlns:tesseract="http://tesseract.com" tesseract:make_convex="true">
<link name="base"/><link name="xlink"/><link name="ylink"/><link name="tcp"/>
<joint name="x" type="prismatic"><parent link="base"/><child link="xlink"/><axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>
<joint name="y" type="prismatic"><parent link="xlink"/><child link="ylink"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>
<joint name="yaw" type="revolute"><parent link="ylink"/><child link="tcp"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint></robot>"""
SRDF = '<robot name="line"><group name="moving"><chain base_link="base" tip_link="tcp"/></group></robot>'


def test_cartesian_line_and_progress_bindings_use_actual_spline_state():
    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    nodes = []
    for i in range(7):
        node = ti.Node(f"knot{i}")
        node.addVar("position", ["x", "y", "yaw"], [i / 6, 0.2, 0.3], [ti.Bounds(-3, 3)] * 3)
        nodes.append(node)
    variables = ti.NodesVariables("knots", nodes)
    knots = [x.getVar("position") for x in variables.getNodes()]
    info = ti.CartLineInfo()
    info.manip = env.getJointGroup("moving")
    info.source_frame = "tcp"
    info.target_frame = "base"
    info.source_frame_offset = Isometry3d.Identity()
    info.target_frame_offset1 = Isometry3d.Identity()
    end = np.eye(4)
    end[0, 3] = 1.0
    info.target_frame_offset2 = Isometry3d(end)
    info.indices = np.arange(6, dtype=np.int32)
    v = np.array([1 / 3, 0.0, 0.0])
    z = np.zeros(3)
    model = native.QuinticSplineModel.build(np.linspace(0, 3, 7).tolist(), v, z, v, z)
    mapping = model.map([0.0, 1.5, 3.0], 0)
    line = native.SplineCartesianLineConstraint(
        info, knots, mapping.basis[1], mapping.offset[1], np.ones(6)
    )
    line.linkWithVariables(variables)
    progress = native.SplineCartesianProgressConstraint(
        info, knots, mapping.basis, mapping.offset, True, True
    )
    progress.linkWithVariables(variables)
    np.testing.assert_allclose(line.getValues(), [0, 0.2, 0, 0, 0, 0.3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        progress.getValues(), [0, 0.5, 1.0, 0, 0.5, 1.0, 0.5, 0.5], atol=1e-10, rtol=0
    )
    assert line.getJacobian().shape == (6, 21)
    assert progress.getJacobian().shape == (8, 21)


def test_fixed_configuration_reversal_has_exact_linear_velocity_jacobian():
    nodes = []
    for i in range(4):
        node = ti.Node(f"reversal{i}")
        node.addVar("position", ["x", "y"], [i / 4, -i / 8], [ti.Bounds(-3, 3)] * 2)
        nodes.append(node)
    variables = ti.NodesVariables("reversal", nodes)
    knots = [node.getVar("position") for node in variables.getNodes()]
    basis = np.array([0.1, -0.3, 0.4, -0.2])
    offset = np.array([0.02, -0.05])
    gradient = np.array([2.0, -1.0])
    term = native.SplineFixedJoinProgressVelocity(knots, basis, offset, gradient)
    term.linkWithVariables(variables)
    original = np.asarray(variables.getValues()).copy()
    expected = gradient @ (basis @ original.reshape(4, 2) + offset)
    np.testing.assert_allclose(term.getValues(), [expected], atol=1e-14, rtol=0)
    jacobian = term.getJacobian().toarray()[0]
    np.testing.assert_allclose(jacobian, np.outer(basis, gradient).ravel(), atol=1e-14, rtol=0)
    step = 1e-6
    for index in range(len(original)):
        candidate = original.copy()
        candidate[index] += step
        variables.setVariables(candidate)
        positive = term.getValues()[0]
        candidate[index] -= 2 * step
        variables.setVariables(candidate)
        negative = term.getValues()[0]
        np.testing.assert_allclose((positive - negative) / (2 * step), jacobian[index], atol=1e-9)
    variables.setVariables(original)


def test_cartesian_line_rejects_knot_axes_different_from_group():
    import pytest

    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    info = ti.CartLineInfo()
    info.manip = env.getJointGroup("moving")
    info.source_frame = "tcp"
    info.target_frame = "base"
    info.source_frame_offset = Isometry3d.Identity()
    info.target_frame_offset1 = Isometry3d.Identity()
    end = np.eye(4)
    end[0, 3] = 1.0
    info.target_frame_offset2 = Isometry3d(end)
    info.indices = np.arange(6, dtype=np.int32)
    nodes = []
    for i in range(2):
        node = ti.Node(f"bad-axes-{i}")
        node.addVar("position", ["x", "y"], [0.0, 0.0], [ti.Bounds(-3, 3)] * 2)
        nodes.append(node)
    variables = ti.NodesVariables("bad-axes", nodes)
    knots = [node.getVar("position") for node in variables.getNodes()]
    with pytest.raises(ValueError, match="kinematics"):
        native.SplineCartesianLineConstraint(info, knots, np.ones(2), np.zeros(2), np.ones(6))
