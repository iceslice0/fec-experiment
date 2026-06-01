PYTHON ?= python3

PROJECT_ROOT := $(CURDIR)
SRC_DIR := $(PROJECT_ROOT)/src
OUTPUT_DIR ?= outputs
SYNTHETIC_OUT ?= $(OUTPUT_DIR)/synthetic
INVERSION_OUT ?= $(OUTPUT_DIR)/inversion
CLIP_OUT ?= $(OUTPUT_DIR)/clip

IMAGENET_DIR ?= ../data/imagenet
REGNET_MODEL ?= regnet_x_3_2gf
CLIP_MODEL ?= ViT-B-32
CLIP_PRETRAINED ?= openai
SKIP_MODEL_DOWNLOAD ?= 0
QS ?= 0.1 0.2 0.3 0.4
MS ?= 1 8 16

INVERSION_CLASS_IDS ?= 24,79,409,701,712,850,950,953,954
INVERSION_CLASS_SLUG ?= 0024_0079_0409_0701_0712_0850_0950_0953_0954
MONTAGE ?= $(INVERSION_OUT)/best_orig_vs_recon_$(INVERSION_CLASS_SLUG).png

BERNOULLI_ARGS ?=
INVERSION_ARGS ?= --class-id $(INVERSION_CLASS_IDS) --batch-size 48 --steps 4000 --lr 0.1 --beta 4 --tv-weight 5e-5 --l2-weight 10.0 --select-best-n 9 --nrow 3
CLIP_ARGS ?=

export PYTHONPATH := $(SRC_DIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.PHONY: install download-models experiments experiment-synthetic experiment-inversion experiment-clip smoke clean

install:
	$(PYTHON) -m pip install -r requirements.txt

download-models:
	$(PYTHON) -m is_fec_experiments.download_models \
		--regnet-model $(REGNET_MODEL) \
		--clip-model $(CLIP_MODEL) \
		--clip-pretrained $(CLIP_PRETRAINED)

experiments: download-models
	$(MAKE) experiment-synthetic
	$(MAKE) experiment-inversion SKIP_MODEL_DOWNLOAD=1
	$(MAKE) experiment-clip SKIP_MODEL_DOWNLOAD=1

ifeq ($(SKIP_MODEL_DOWNLOAD),0)
experiment-inversion: download-models
experiment-clip: download-models
endif

experiment-synthetic:
	QS="$(QS)" MS="$(MS)" LOG_DIR="$(PROJECT_ROOT)/$(SYNTHETIC_OUT)/logs" \
		bash scripts/run_synthetic_grid.sh $(BERNOULLI_ARGS)

experiment-inversion:
	mkdir -p "$(INVERSION_OUT)"
	$(PYTHON) -m is_fec_experiments.inversion.invert_batch_imagenet_val \
		--data-dir "$(IMAGENET_DIR)" \
		--model "$(REGNET_MODEL)" \
		--out "$(INVERSION_OUT)" \
		$(INVERSION_ARGS)

experiment-clip:
	mkdir -p "$(CLIP_OUT)"
	$(PYTHON) -m is_fec_experiments.inversion.clip_montage_eval \
		"$(MONTAGE)" \
		--model "$(CLIP_MODEL)" \
		--pretrained "$(CLIP_PRETRAINED)" \
		--output-json "$(CLIP_OUT)/clip_montage_metrics.json" \
		--output-csv "$(CLIP_OUT)/clip_montage_per_pair.csv" \
		$(CLIP_ARGS)

smoke:
	$(PYTHON) -m compileall -q src

clean:
	rm -rf build dist *.egg-info
	rm -rf outputs
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
