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

## Smoke Check

For a fast local check that does not download models or require ImageNet:

```bash
make smoke
```

This compiles the package without downloading models or requiring ImageNet.
