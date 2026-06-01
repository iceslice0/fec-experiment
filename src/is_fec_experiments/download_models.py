#!/usr/bin/env python3
"""Download pretrained models used by the experiments."""

from __future__ import annotations

import argparse

from is_fec_experiments.model_registry import (
    available_regnet_models,
    load_regnet_model,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download/cache pretrained Torchvision and OpenCLIP models."
    )
    parser.add_argument(
        "--regnet-model",
        default="regnet_x_3_2gf",
        choices=available_regnet_models(),
        help="Torchvision RegNet model to cache. Default: regnet_x_3_2gf.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="OpenCLIP model name to cache. Default: ViT-B-32.",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="openai",
        help="OpenCLIP pretrained weights tag. Default: openai.",
    )
    parser.add_argument(
        "--skip-regnet",
        action="store_true",
        help="Do not download Torchvision RegNet weights.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Do not download OpenCLIP weights.",
    )
    args = parser.parse_args(argv)

    if not args.skip_regnet:
        print(f"Caching Torchvision model: {args.regnet_model}")
        model = load_regnet_model(args.regnet_model)
        model.eval()
        print("Torchvision model cached.")

    if not args.skip_clip:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: open_clip_torch. Install project dependencies first."
            ) from exc

        print(
            f"Caching OpenCLIP model: {args.clip_model} "
            f"({args.clip_pretrained})"
        )
        model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
        )
        model.eval()
        print("OpenCLIP model cached.")


if __name__ == "__main__":
    main()
