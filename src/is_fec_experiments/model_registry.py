"""Torchvision model helpers used by the inversion experiment."""

from __future__ import annotations

REGNET_MODEL_SPECS = {
    "regnet_y_3_2gf": ("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights"),
    "regnet_x_3_2gf": ("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights"),
}


def available_regnet_models() -> tuple[str, ...]:
    return tuple(sorted(REGNET_MODEL_SPECS))


def load_regnet_model(name: str):
    """Load a pretrained RegNet model, triggering Torchvision's weight download."""
    if name not in REGNET_MODEL_SPECS:
        choices = ", ".join(available_regnet_models())
        raise ValueError(f"Unknown RegNet model '{name}'. Choices: {choices}")

    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: torchvision. Install project dependencies first."
        ) from exc

    factory_name, weights_name = REGNET_MODEL_SPECS[name]
    factory = getattr(models, factory_name)
    weights = getattr(models, weights_name).DEFAULT
    return factory(weights=weights)
