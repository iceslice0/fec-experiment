# IS-FEC Experiments

Companion experiment code for the paper on interpreting neural classifiers as forward-error-correcting decoders.

## Project Layout

```text
src/is_fec_experiments/
  synthetic/      Bernoulli-channel linear softmax experiments
  inversion/      ImageNet RegNet feature inversion and CLIP montage scoring
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

`make download-models` caches the Torchvision RegNet weights and the OpenCLIP weights. The default inversion model is `regnet_x_3_2gf`, matching the original script behavior. ImageNet is expected at `../data/imagenet` by default.

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
make experiment-clip
```

Useful overrides:

```bash
make experiment-inversion IMAGENET_DIR=../data/imagenet REGNET_MODEL=regnet_x_3_2gf
make experiment-synthetic QS="0.1 0.2" MS="1 8" BERNOULLI_ARGS="--max_epochs 5"
make experiment-clip MONTAGE=outputs/inversion/best_orig_vs_recon_0024_0079_0409_0701_0712_0850_0950_0953_0954.png
```

Outputs are written under `outputs/synthetic`, `outputs/inversion`, and `outputs/clip`.

## Smoke Check

For a fast local check that does not download models or require ImageNet:

```bash
make smoke
```

This compiles the package without downloading models or requiring ImageNet.
