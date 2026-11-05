"""The driver uses one scene and never marks sampled reports as accepted."""

import json
import sys

import pytest
import run_uniform_sweep as driver


def inputs(tmp_path):
    station = tmp_path / "station"
    (station / "programs").mkdir(parents=True)
    (station / "programs" / "weld_trajectory.json").write_text("{}")
    (station / "joint_speeds.json").write_text('{"joint_speeds": {"joints": {}}}')
    original = tmp_path / "original"
    original.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name in ("fresh.json", "verified.json", "fixture.json", "ifopt.dylib", "sqp.dylib"):
        (tmp_path / name).write_text("{}")
    return [
        "sweep",
        "--station",
        str(station),
        "--original-worktree",
        str(original),
        "--fresh-route",
        str(tmp_path / "fresh.json"),
        "--verified-route",
        str(tmp_path / "verified.json"),
        "--artifacts",
        str(artifacts),
        "--fixture-policy",
        str(tmp_path / "fixture.json"),
        "--output",
        str(tmp_path / "output"),
        "--ifopt-library",
        str(tmp_path / "ifopt.dylib"),
        "--sqp-library",
        str(tmp_path / "sqp.dylib"),
        "--factors",
        "1",
        "2",
    ]


def test_one_scene_and_explicit_sampled_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", inputs(tmp_path))
    monkeypatch.setattr(driver, "require_loaded_libraries", lambda paths: paths)
    scene = {"inputs": tuple(range(12))}
    calls = []
    monkeypatch.setattr(driver, "initialize_scene", lambda *_: calls.append("scene") or scene)
    monkeypatch.setattr(driver, "prepare_fresh", lambda *_: ([], {"stages": []}, ()))
    monkeypatch.setattr(
        driver,
        "solve_duration_policy",
        lambda factor, got_scene, *_: calls.append(("solve", factor, got_scene is scene))
        or {"elapsed_s": 1.0},
    )
    monkeypatch.setattr(
        driver,
        "verify_duration_policy",
        lambda factor, got_scene, *_: calls.append(("check", factor, got_scene is scene))
        or {
            "elapsed_s": 2.0,
            "modeled_tour_duration_s": 3.0,
            "stage_count": 67,
            "checked_samples": 100,
            "negative_contacts": 0,
            "raw_limit_excess": [0.0] * 9,
        },
    )
    driver.main()
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert calls == [
        "scene",
        ("solve", 1.0, True),
        ("check", 1.0, True),
        ("solve", 2.0, True),
        ("check", 2.0, True),
    ]
    assert all(policy["overall_accepted"] is False for policy in manifest["policies"])


def test_duplicate_factors_fail_before_scene_setup(tmp_path, monkeypatch):
    argv = inputs(tmp_path)
    argv[-1] = "1"
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        driver, "initialize_scene", lambda *_: pytest.fail("scene should not be loaded")
    )
    with pytest.raises(RuntimeError, match="unique"):
        driver.main()
