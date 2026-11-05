"""Exercise a coordinated SearchL and return on a real synthetic FK model."""

import numpy as np
from process_horizon import build_process_horizon
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp
from tesseract_robotics.tesseract_common import GeneralResourceLocator, Isometry3d
from tesseract_robotics.tesseract_environment import Environment

URDF = """<robot name="line" xmlns:tesseract="http://tesseract.com" tesseract:make_convex="true">
<link name="base"/><link name="xlink"/><link name="ylink"/><link name="tcp"/>
<joint name="x" type="prismatic"><parent link="base"/><child link="xlink"/><axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>
<joint name="y" type="prismatic"><parent link="xlink"/><child link="ylink"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>
<joint name="yaw" type="revolute"><parent link="ylink"/><child link="tcp"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint></robot>"""
SRDF = '<robot name="line"><group name="moving"><chain base_link="base" tip_link="tcp"/></group></robot>'


def test_coordinated_search_return_keeps_endpoints_order_and_smooths():
    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    group = env.getJointGroup("moving")

    def line(start, end):
        info = ti.CartLineInfo()
        info.manip = group
        info.source_frame = "tcp"
        info.target_frame = "base"
        info.source_frame_offset = Isometry3d.Identity()
        a = np.eye(4)
        a[0, 3] = start
        b = np.eye(4)
        b[0, 3] = end
        info.target_frame_offset1 = Isometry3d(a)
        info.target_frame_offset2 = Isometry3d(b)
        info.indices = np.arange(6, dtype=np.int32)
        return info

    times = np.linspace(0, 2, 9)
    u = times / 2
    seed = np.zeros((9, 3))
    seed[:, 0] = 16 * u**2 * (1 - u) ** 2
    seed[:, 1] = 0.08 * np.sin(np.pi * u)
    legs = [
        {"line": line(0.0, 1.0), "start_s": 0.0, "end_s": 1.0},
        {"line": line(1.0, 0.0), "start_s": 1.0, "end_s": 2.0},
    ]
    variables, problem, model = build_process_horizon(
        ["x", "y", "yaw"],
        times,
        seed,
        np.tile([-3.0, 3.0], (3, 1)),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.ones(3),
        legs,
        5,
        lambda *_: [],
        fixed_reversals=[(4, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))],
    )
    assert "fixed-reversal-4" in problem.getNLPConstraintNames()
    join_bounds = variables.getBounds()[12:15]
    assert all(
        b.getLower() == b.getUpper() == value for b, value in zip(join_bounds, [1.0, 0.0, 0.0])
    )
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    settings = solver.params
    settings.max_iter = 200
    settings.max_time = 10.0
    solver.params = settings
    solver.solve(problem)
    assert solver.getStatus() == sqp.SQPStatus.NLP_CONVERGED
    q = np.asarray(variables.getValues()).reshape(9, 3)
    sampled = model.evaluate(np.linspace(0, 2, 101), q, 1)
    assert np.max(np.abs(sampled[0][:, 1])) < 1e-4
    np.testing.assert_allclose(sampled[0][[0, 50, 100], 0], [0, 1, 0], atol=1e-4, rtol=0)
    assert abs(sampled[1][50, 0]) < 1e-3


def test_explicit_soft_line_and_fixed_axis_keep_progress_hard():
    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    group = env.getJointGroup("moving")
    info = ti.CartLineInfo()
    info.manip = group
    info.source_frame = "tcp"
    info.target_frame = "base"
    info.source_frame_offset = Isometry3d.Identity()
    info.target_frame_offset1 = Isometry3d.Identity()
    b = np.eye(4)
    b[0, 3] = 1.0
    info.target_frame_offset2 = Isometry3d(b)
    info.indices = np.arange(6, dtype=np.int32)
    times = np.linspace(0.0, 1.0, 7)
    seed = np.zeros((7, 3))
    seed[:, 0] = times
    variables, problem, _ = build_process_horizon(
        ["x", "y", "yaw"],
        times,
        seed,
        np.tile([-3.0, 3.0], (3, 1)),
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        np.ones(3),
        [{"line": info, "start_s": 0.0, "end_s": 1.0}],
        3,
        lambda *_: [],
        line_mode="soft",
        fixed_axes={1: 0.0},
        line_witnesses=[(0, 0.25)],
    )
    assert sum(name.startswith("line-") for name in problem.getNLPCostNames()) == 4
    assert any(name.startswith("progress-") for name in problem.getNLPConstraintNames())
    bounds = variables.getBounds()
    assert all(
        bounds[i * 3 + 1].getLower() == bounds[i * 3 + 1].getUpper() == 0.0 for i in range(7)
    )


def test_fixed_axis_normalizes_only_machine_precision_seed_residual():
    import pytest
    from process_horizon import ProcessHorizonError

    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    group = env.getJointGroup("moving")
    info = ti.CartLineInfo()
    info.manip = group
    info.source_frame = "tcp"
    info.target_frame = "base"
    info.source_frame_offset = Isometry3d.Identity()
    info.target_frame_offset1 = Isometry3d.Identity()
    b = np.eye(4)
    b[0, 3] = 1.0
    info.target_frame_offset2 = Isometry3d(b)
    info.indices = np.arange(6, dtype=np.int32)
    times = np.linspace(0.0, 1.0, 7)
    seed = np.zeros((7, 3))
    seed[:, 0] = times
    seed[3, 1] = np.finfo(float).eps

    def build(q):
        return build_process_horizon(
            ["x", "y", "yaw"],
            times,
            q,
            np.tile([-3.0, 3.0], (3, 1)),
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            np.ones(3),
            [{"line": info, "start_s": 0.0, "end_s": 1.0}],
            3,
            lambda *_: [],
            fixed_axes={1: 0.0},
        )

    variables, _, _ = build(seed)
    assert variables.getValues()[10] == 0.0
    seed[3, 1] = 0.01
    with pytest.raises(ProcessHorizonError):
        build(seed)
