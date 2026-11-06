"""The optional jerk pass consumes the selected independently verified candidate."""

import pytest
from fresh_product_jerk import FreshProductJerkError, run_verified_product


def test_postpass_receives_verified_candidate_in_same_scene():
    scene = {"name": "one retained scene"}
    record = {"result": {"path": "fresh"}}
    checked = {"status": "independently_sampled_tour", "expanded": {"segments": []}}
    calls = []

    def verify(*args, **kwargs):
        assert args[-1] is record
        return checked

    def product(given, tour, *, verify, **kwargs):
        assert given is scene and tour == 20
        assert verify("scene-input", record) is checked
        return {"status": "independently_sampled_tour"}

    def postpass(given, tour, selected, verification):
        calls.append((given, tour, selected, verification))
        return {"smoothness_gate": "unresolved"}

    result = run_verified_product(scene, 20, run_product=product, verify=verify, postpass=postpass)
    assert calls == [(scene, 20, record, checked)]
    assert result["jerk"]["smoothness_gate"] == "unresolved"


def test_failed_product_cannot_enter_postpass():
    def product(*args, **kwargs):
        return {"status": "wall_budget_exhausted"}

    def fail(*args, **kwargs):
        raise AssertionError("must not run")

    with pytest.raises(FreshProductJerkError, match="not verified"):
        run_verified_product({}, 9, run_product=product, verify=fail, postpass=fail)


def test_success_without_captured_verified_candidate_fails_loudly():
    def product(*args, **kwargs):
        return {"status": "independently_sampled_tour"}

    with pytest.raises(FreshProductJerkError, match="no verified candidate"):
        run_verified_product(
            {}, 9, run_product=product, verify=lambda *a, **k: None, postpass=lambda *a: None
        )
