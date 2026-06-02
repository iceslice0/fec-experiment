# IS-FEC Experiments

Companion experiment code for the paper on interpreting neural classifiers as forward-error-correcting decoders.

## Project Layout

```text
src/is_fec_experiments/
  synthetic/      Bernoulli-channel linear softmax experiments
  inversion/      ImageNet feature inversion and CLIP montage scoring
scripts/          Batch experiment runners
outputs/          Generated logs, reconstructions, and metrics
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
make install
pip install -e .
make download-models
```

`make download-models` caches the Torchvision model weights and the OpenCLIP weights. The default inversion model is `regnet_x_3_2gf`, matching the original script behavior. Available models are `regnet_x_400mf`, `regnet_y_400mf`, `regnet_x_800mf`, `regnet_y_800mf`, `regnet_x_1_6gf`, `regnet_y_1_6gf`, `regnet_x_3_2gf`, `regnet_y_3_2gf`, `regnet_x_8gf`, `regnet_y_8gf`, `regnet_x_16gf`, `regnet_y_16gf`, `regnet_x_32gf`, `regnet_y_32gf`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, and `resnet152`. ImageNet is expected at `../data/imagenet` by default.

The pinned Torch/Torchvision versions in `requirements.txt` match the environment used for the reference scripts in `/home/iceslice/nn/gen`.

## Run Experiments

Run the full experiment pipeline:

```bash
make experiments
```

Individual experiment targets:

```bash
make experiment-synthetic
make experiment-inversion
make experiment-inversion48
make experiment-inversion-all
make experiment-inversion48-all
make experiment-inversions-all
make experiment-clip
make experiment-clip48
make experiment-clip-all
make experiment-clip48-all
make experiment-clips-all
```

Useful overrides:

```bash
make experiment-inversion IMAGENET_DIR=../data/imagenet MODEL=resnet50
make experiment-inversion-regnet_x_400mf
make experiment-inversion-regnet_x_8gf
make experiment-inversion-resnet18
make experiment-inversion-resnet34
make experiment-inversion-resnet50
make experiment-inversion-resnet101
make experiment-inversion-resnet152
make experiment-clip-regnet_y_800mf
make experiment-clip-resnet50
make experiment-synthetic QS="0.1 0.2" MS="1 8" BERNOULLI_ARGS="--max_epochs 5"
make experiment-clip MONTAGE=outputs/inversion/regnet_x_3_2gf/best_orig_vs_recon_0024_0079_0409_0701_0712_0850_0950_0953_0954.png
```

Each model has separate `experiment-inversion-<model>`, `experiment-inversion48-<model>`, `experiment-clip-<model>`, and `experiment-clip48-<model>` targets.

Outputs are written under `outputs/synthetic`, model-specific `outputs/inversion/<model>`, `outputs/inversion48/<model>`, `outputs/clip/<model>`, and `outputs/clip48/<model>`.

Inversion runs also write final metrics, initial KL, and classification accuracy to `metrics.json` in the inversion output directory.

## Inversion Overview

The table below summarizes the current 48-image inversion runs from `outputs/inversion48/<model>/metrics.json` and normalized CLIP montage results from `outputs/clip48/<model>/clip_montage_metrics.json`; models without both files are omitted.

`Inv acc` is reconstructed-image top-1 classifier accuracy from the inversion model family. `KL before` and `KL after` are the average symmetric KL values recorded before optimization and at the final reconstruction. CLIP columns use the row/column-normalized `S_norm` block: top-1/top-5 retrieval recall, mean matched similarity, and mean random mismatched similarity.

| Model | Inv acc | KL before | KL after | CLIP top-1 | CLIP top-5 | CLIP matched | CLIP mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `regnet_x_400mf` | 95.8% | 224.80 | 7.74 | 33.3% | 72.9% | 0.662 | 0.605 |
| `regnet_y_400mf` | 93.8% | 262.62 | 12.13 | 31.2% | 56.2% | 0.672 | 0.613 |
| `regnet_x_800mf` | 91.7% | 257.03 | 26.50 | 33.3% | 52.1% | 0.678 | 0.623 |
| `regnet_y_800mf` | 100.0% | 343.05 | 14.57 | 31.2% | 66.7% | 0.626 | 0.587 |
| `regnet_x_1_6gf` | 97.9% | 243.25 | 22.92 | 27.1% | 75.0% | 0.660 | 0.612 |
| `regnet_y_1_6gf` | 97.9% | 243.96 | 17.37 | 31.2% | 47.9% | 0.639 | 0.607 |
| `regnet_x_3_2gf` | 100.0% | 234.93 | 16.00 | 43.8% | 56.2% | 0.658 | 0.603 |
| `regnet_y_3_2gf` | 97.9% | 230.04 | 12.77 | 22.9% | 58.3% | 0.612 | 0.576 |
| `regnet_x_8gf` | 83.3% | 224.15 | 56.10 | 33.3% | 68.8% | 0.639 | 0.597 |
| `regnet_y_8gf` | 89.6% | 193.35 | 27.12 | 8.3% | 50.0% | 0.612 | 0.584 |
| `resnet18` | 10.4% | 412.46 | 160.66 | 6.2% | 25.0% | 0.553 | 0.545 |
| `resnet34` | 0.0% | 1001.83 | 870.37 | 0.0% | 8.3% | 0.539 | 0.538 |
| `resnet50` | 93.8% | 226.96 | 36.75 | 29.2% | 58.3% | 0.640 | 0.597 |
| `resnet101` | 75.0% | 250.69 | 79.55 | 14.6% | 39.6% | 0.618 | 0.589 |
| `resnet152` | 85.4% | 218.18 | 56.39 | 20.8% | 45.8% | 0.624 | 0.588 |

## Smoke Check

For a fast local check that does not download models or require ImageNet:

```bash
make smoke
```

This compiles the package without downloading models or requiring ImageNet.
