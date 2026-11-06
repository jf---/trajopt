"""Opt-in jerk pass after a fresh product candidate is independently verified."""

from collections.abc import Callable
from typing import Any


class FreshProductJerkError(RuntimeError):
    """A fresh candidate is not available for the requested jerk pass."""


def run_verified_product(
    scene: dict[str, Any],
    tour: int,
    *,
    run_product: Callable[..., dict[str, Any]],
    verify: Callable[..., dict[str, Any]],
    postpass: Callable[[dict[str, Any], int, dict[str, Any], dict[str, Any]], dict[str, Any]],
    **product_options: Any,
) -> dict[str, Any]:
    """Run the existing product path and postprocess its selected verified result."""
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = verify(*args, **kwargs)
        if result.get("status") == "independently_sampled_tour":
            record = args[-1]
            if not isinstance(record, dict):
                raise FreshProductJerkError("verified candidate record is malformed")
            selected.append((record, result))
        return result

    product = run_product(scene, tour, verify=capture, **product_options)
    if product.get("status") != "independently_sampled_tour":
        raise FreshProductJerkError(f"fresh tour not verified: {product.get('status')}")
    if not selected:
        raise FreshProductJerkError("product returned no verified candidate")
    record, verified = selected[-1]
    return {"product": product, "jerk": postpass(scene, tour, record, verified)}
