"""Exercise the complete verifier with a tiny, deterministic fake nine-axis cell."""

import json
from types import SimpleNamespace

import numpy as np
import verify_duration_policy as verifier


def test_all_67_stages_are_checked_without_claiming_physical_acceptance(tmp_path, monkeypatch):
    labels = ["transit:0"]
    for i in range(3):
        labels += [
            f"approach:{i}",
            f"search:{i}",
            f"retract:{i}",
            f"withdraw:{i}",
            f"transit:{i + 1}",
        ]
    labels += [f"weld:{i}" for i in range(4, 10)] + ["transit:10"]
    labels += [f"fixture:{i}" for i in range(11, 16)] + ["transit:16"]
    for i in range(3, 6):
        labels += [
            f"approach:{i}",
            f"search:{i}",
            f"retract:{i}",
            f"withdraw:{i}",
            f"transit:{i + 14}",
        ]
    labels += ["weld:20", "weld:21", "weld:22", "transit:23"]
    for i in range(6, 9):
        labels += [
            f"approach:{i}",
            f"search:{i}",
            f"retract:{i}",
            f"withdraw:{i}",
            f"transit:{i + 18}",
        ]
    labels += ["weld:27", "weld:28", "weld:29", "transit:30"]
    assert len(labels) == 67
    stages = []
    x = 0.0
    for label in labels:
        end = x if label.startswith("fixture:") else x + 1.0
        start_state = [x] + [0.0] * 8
        end_state = [end] + [0.0] * 8
        stages.append({"stage": label, "states": [start_state, end_state]})
        x = end
    stage_by_name = {stage["stage"]: stage for stage in stages}
    clock = {"stages": [{"stage": label, "duration_s": 1.0} for label in labels]}
    artifacts = tmp_path / "artifacts"
    for index in range(9):
        folder = artifacts / "all-probes-pass1" / f"probe{index:02}"
        folder.mkdir(parents=True)
        values = [
            [row[:7] for row in stage_by_name[f"{phase}:{index}"]["states"]]
            for phase in ("approach", "search", "retract", "withdraw")
        ]
        (folder / "ik-progress-report.json").write_text(json.dumps({"stage_states": values}))
    for numbers, folder in (
        (range(4, 10), artifacts),
        (range(20, 23), artifacts / "local-weld-20"),
        (range(27, 30), artifacts / "local-weld-27"),
    ):
        folder.mkdir(parents=True, exist_ok=True)
        names = [f"weld:{number}" for number in numbers]
        points = [stage_by_name[names[0]]["states"][0][:7]] + [
            stage_by_name[name]["states"][-1][:7] for name in names
        ]
        data = {
            "joint_times_s": list(range(len(points))),
            "joint_states": points,
            "left_blend_windows_s": [0.0] * len(names),
            "right_blend_windows_s": [0.0] * len(names),
        }
        (folder / "weld-local-extent-1.json").write_text(json.dumps(data))
    fixture_start = stage_by_name["fixture:11"]["states"][0]
    fixture = {
        "fixture": {
            "start": fixture_start,
            "end": fixture_start,
            "stage_times_s": [0, 1, 2, 3, 4, 5],
            "duration_s": 5.0,
        }
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    output = tmp_path / "output"
    folder = output / "global-duration-1"
    folder.mkdir(parents=True)
    transit_results = []
    for stage in stages:
        if stage["stage"].startswith("transit:"):
            q = np.asarray(stage["states"])
            curve = {
                "stage": stage["stage"],
                "duration_s": 1.0,
                "knots_s": [0.0, 1.0],
                "nodes": q[:, :7].tolist(),
                "v0": [0.0] * 7,
                "a0": [0.0] * 7,
                "v1": [0.0] * 7,
                "a1": [0.0] * 7,
                "fixture": [0.0, 0.0],
            }
            transit_results.append(
                {"stage": stage["stage"], "status": "sampled_validated", "curve": curve}
            )
    (folder / "summary.json").write_text(json.dumps({"factor": 1.0, "transits": transit_results}))

    class FakeModel:
        @staticmethod
        def build(*_args):
            return FakeModel()

        def evaluate(self, times, nodes, _order):
            t = np.asarray(times)[:, None]
            nodes = np.asarray(nodes)
            pos = nodes[0] + t * (nodes[-1] - nodes[0])
            return pos, np.tile(nodes[-1] - nodes[0], (len(t), 1)), np.zeros_like(pos)

        def map(self, times, _order):
            return SimpleNamespace(
                basis=np.zeros((len(times), 2)), offset=np.zeros((len(times), 7))
            )

    class FakeEnv:
        def getState(self, values=None):
            transform = np.eye(4)
            transform[0, 3] = (values or {}).get("track", 0.0)
            links = {
                "workpiece_link": SimpleNamespace(matrix=np.eye(4)),
                "nozzle_link": SimpleNamespace(matrix=transform),
            }
            return SimpleNamespace(joints={}, link_transforms=links)

        def getRevision(self):
            return 0

        def getJointGroup(self, *_args):
            return SimpleNamespace(
                getLimits=lambda: SimpleNamespace(joint_limits=np.tile([-100.0, 100.0], (9, 1)))
            )

    class FakeChecker:
        @staticmethod
        def build(*_args):
            return SimpleNamespace(contacts=lambda _state: [])

    monkeypatch.setattr(verifier, "native", SimpleNamespace(QuinticSplineModel=FakeModel))
    monkeypatch.setattr(verifier, "StateChecker", FakeChecker)
    monkeypatch.setattr(verifier, "snapshot", lambda _scene: "unchanged")
    monkeypatch.setattr(verifier, "_restore_scene", lambda *_args: None)
    monkeypatch.setattr(verifier, "TRANSIT_SAMPLES", 3)
    monkeypatch.setattr(verifier, "PROCESS_SAMPLES", 3)
    monkeypatch.setattr(verifier, "LOCAL_WINDOW_SAMPLES", 3)
    names = ("track", "a", "b", "c", "d", "e", "f", "positioner", "tailstock")
    env = FakeEnv()
    scene = {"inputs": (env, None, None, None, names, None, None, None, None, None, None, None)}
    report = verifier.verify_duration_policy(
        1.0, scene, stages, clock, np.ones(7), output, artifacts, fixture_path
    )
    assert report["stage_count"] == 67
    assert report["checked_samples"] == 201
    assert len(report["joins"]) == 66
    assert sum(stage["process_geometry"] is not None for stage in report["stages"]) == 48
    assert report["negative_contacts"] == 0
    assert report["overall_accepted"] is False
    assert report["stages"][0]["max_jerk_by_axis"] == [0.0] * 9
