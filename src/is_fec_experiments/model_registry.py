"""Torchvision model helpers used by the inversion experiment."""

from __future__ import annotations

MODEL_SPECS = {
    "regnet_x_400mf": ("regnet_x_400mf", "RegNet_X_400MF_Weights", "trunk_output"),
    "regnet_y_400mf": ("regnet_y_400mf", "RegNet_Y_400MF_Weights", "trunk_output"),
    "regnet_x_800mf": ("regnet_x_800mf", "RegNet_X_800MF_Weights", "trunk_output"),
    "regnet_y_800mf": ("regnet_y_800mf", "RegNet_Y_800MF_Weights", "trunk_output"),
    "regnet_x_1_6gf": ("regnet_x_1_6gf", "RegNet_X_1_6GF_Weights", "trunk_output"),
    "regnet_y_1_6gf": ("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights", "trunk_output"),
    "regnet_x_3_2gf": ("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights", "trunk_output"),
    "regnet_y_3_2gf": ("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights", "trunk_output"),
    "regnet_x_8gf": ("regnet_x_8gf", "RegNet_X_8GF_Weights", "trunk_output"),
    "regnet_y_8gf": ("regnet_y_8gf", "RegNet_Y_8GF_Weights", "trunk_output"),
    "regnet_x_16gf": ("regnet_x_16gf", "RegNet_X_16GF_Weights", "trunk_output"),
    "regnet_y_16gf": ("regnet_y_16gf", "RegNet_Y_16GF_Weights", "trunk_output"),
    "regnet_x_32gf": ("regnet_x_32gf", "RegNet_X_32GF_Weights", "trunk_output"),
    "regnet_y_32gf": ("regnet_y_32gf", "RegNet_Y_32GF_Weights", "trunk_output"),
    "resnet18": ("resnet18", "ResNet18_Weights", "layer4"),
    "resnet34": ("resnet34", "ResNet34_Weights", "layer4"),
    "resnet50": ("resnet50", "ResNet50_Weights", "layer4"),
    "resnet101": ("resnet101", "ResNet101_Weights", "layer4"),
    "resnet152": ("resnet152", "ResNet152_Weights", "layer4"),
}


def available_models() -> tuple[str, ...]:
    return tuple(MODEL_SPECS)


def load_model(name: str):
    """Load a pretrained Torchvision model, triggering its weight download."""
    if name not in MODEL_SPECS:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{name}'. Choices: {choices}")

    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: torchvision. Install project dependencies first."
        ) from exc

    factory_name, weights_name, _ = MODEL_SPECS[name]
    factory = getattr(models, factory_name)
    weights = getattr(models, weights_name).DEFAULT
    return factory(weights=weights)


def feature_module_name(name: str) -> str:
    if name not in MODEL_SPECS:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{name}'. Choices: {choices}")
    return MODEL_SPECS[name][2]


def get_feature_module(model, name: str):
    return getattr(model, feature_module_name(name))


def available_regnet_models() -> tuple[str, ...]:
    return tuple(name for name in MODEL_SPECS if name.startswith("regnet_"))


def load_regnet_model(name: str):
    if name not in available_regnet_models():
        choices = ", ".join(available_regnet_models())
        raise ValueError(f"Unknown RegNet model '{name}'. Choices: {choices}")
    return load_model(name)
