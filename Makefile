PYTHON ?= python3

PROJECT_ROOT := $(CURDIR)
SRC_DIR := $(PROJECT_ROOT)/src
OUTPUT_DIR ?= outputs
REGNET_MODEL ?= regnet_x_3_2gf
MODEL ?= $(REGNET_MODEL)
REGNET_MODELS ?= regnet_x_400mf regnet_y_400mf regnet_x_800mf regnet_y_800mf regnet_x_1_6gf regnet_y_1_6gf regnet_x_3_2gf regnet_y_3_2gf regnet_x_8gf regnet_y_8gf
MODELS ?= $(REGNET_MODELS) resnet18 resnet34 resnet50 resnet101 resnet152
MODEL_INVERSION_TARGETS := $(addprefix experiment-inversion-,$(MODELS))
MODEL_INVERSION48_TARGETS := $(addprefix experiment-inversion48-,$(MODELS))
MODEL_CLIP_TARGETS := $(addprefix experiment-clip-,$(MODELS))
MODEL_CLIP48_TARGETS := $(addprefix experiment-clip48-,$(MODELS))
SYNTHETIC_OUT ?= $(OUTPUT_DIR)/synthetic
INVERSION_OUT ?= $(OUTPUT_DIR)/inversion/$(MODEL)
INVERSION48_OUT ?= $(OUTPUT_DIR)/inversion48/$(MODEL)
INVERSION48_CLASS_SLUG ?= all
CLIP_OUT ?= $(OUTPUT_DIR)/clip/$(MODEL)
CLIP48_OUT ?= $(OUTPUT_DIR)/clip48/$(MODEL)

IMAGENET_DIR ?= ../data/imagenet
CLIP_MODEL ?= ViT-B-32
CLIP_PRETRAINED ?= openai
SKIP_MODEL_DOWNLOAD ?= 0
QS ?= 0.1 0.2 0.3 0.4
MS ?= 1 8 16

INVERSION_CLASS_IDS ?= 24,79,409,701,712,850,950,953,954
INVERSION_CLASS_SLUG ?= 0024_0079_0409_0701_0712_0850_0950_0953_0954
MONTAGE ?= $(INVERSION_OUT)/best_orig_vs_recon_$(INVERSION_CLASS_SLUG).png
MONTAGE48 ?= $(INVERSION48_OUT)/best_orig_vs_recon_$(INVERSION48_CLASS_SLUG).png

BERNOULLI_ARGS ?=
INVERSION_ARGS ?= --class-id $(INVERSION_CLASS_IDS) --batch-size 45 --select-best-n 9 --nrow 6
INVERSION48_ARGS ?= --batch-size 48 --select-best-n 48 --nrow 8
CLIP_ARGS ?=

export PYTHONPATH := $(SRC_DIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.PHONY: install download-models download-torchvision-model download-regnet-model download-clip-model experiments experiment-synthetic run-inversion experiment-inversion experiment-inversion48 experiment-inversion-all experiment-inversion48-all experiment-inversions-all run-clip experiment-clip experiment-clip48 experiment-clip-all experiment-clip48-all experiment-clips-all $(MODEL_INVERSION_TARGETS) $(MODEL_INVERSION48_TARGETS) $(MODEL_CLIP_TARGETS) $(MODEL_CLIP48_TARGETS) smoke clean

install:
	$(PYTHON) -m pip install -r requirements.txt

download-models:
	$(PYTHON) -m is_fec_experiments.download_models \
		--model $(MODEL) \
		--clip-model $(CLIP_MODEL) \
		--clip-pretrained $(CLIP_PRETRAINED)

download-torchvision-model:
	$(PYTHON) -m is_fec_experiments.download_models \
		--model $(MODEL) \
		--skip-clip

download-regnet-model: download-torchvision-model

download-clip-model:
	$(PYTHON) -m is_fec_experiments.download_models \
		--skip-torchvision \
		--clip-model $(CLIP_MODEL) \
		--clip-pretrained $(CLIP_PRETRAINED)

experiments: download-models
	$(MAKE) experiment-synthetic
	$(MAKE) experiment-inversion SKIP_MODEL_DOWNLOAD=1
	$(MAKE) experiment-clip SKIP_MODEL_DOWNLOAD=1

ifeq ($(SKIP_MODEL_DOWNLOAD),0)
experiment-inversion: download-torchvision-model
experiment-inversion48: download-torchvision-model
experiment-clip: download-clip-model
experiment-clip48: download-clip-model
experiment-clip-all: download-clip-model
experiment-clip48-all: download-clip-model
endif

experiment-synthetic:
	QS="$(QS)" MS="$(MS)" LOG_DIR="$(PROJECT_ROOT)/$(SYNTHETIC_OUT)/logs" \
		bash scripts/run_synthetic_grid.sh $(BERNOULLI_ARGS)

run-inversion:
	mkdir -p "$(INVERSION_OUT)"
	$(PYTHON) -m is_fec_experiments.inversion.invert_batch_imagenet_val \
		--data-dir "$(IMAGENET_DIR)" \
		--model "$(MODEL)" \
		--out "$(INVERSION_OUT)" \
		$(INVERSION_ARGS)

experiment-inversion: run-inversion

experiment-inversion48: INVERSION_OUT := $(INVERSION48_OUT)
experiment-inversion48: INVERSION_ARGS := $(INVERSION48_ARGS)
experiment-inversion48: run-inversion

run-clip:
	mkdir -p "$(CLIP_OUT)"
	$(PYTHON) -m is_fec_experiments.inversion.clip_montage_eval \
		"$(MONTAGE)" \
		--model "$(CLIP_MODEL)" \
		--pretrained "$(CLIP_PRETRAINED)" \
		--output-json "$(CLIP_OUT)/clip_montage_metrics.json" \
		--output-csv "$(CLIP_OUT)/clip_montage_per_pair.csv" \
		$(CLIP_ARGS)

experiment-clip: run-clip

experiment-clip48: CLIP_OUT := $(CLIP48_OUT)
experiment-clip48: MONTAGE := $(MONTAGE48)
experiment-clip48: run-clip

define MODEL_TARGETS
experiment-inversion-$(1):
	$$(MAKE) experiment-inversion MODEL=$(1)

experiment-inversion48-$(1):
	$$(MAKE) experiment-inversion48 MODEL=$(1)

experiment-clip-$(1):
	$$(MAKE) experiment-clip MODEL=$(1)

experiment-clip48-$(1):
	$$(MAKE) experiment-clip48 MODEL=$(1)

endef

$(foreach model,$(MODELS),$(eval $(call MODEL_TARGETS,$(model))))

experiment-inversion-all:
	$(MAKE) $(MODEL_INVERSION_TARGETS)

experiment-inversion48-all:
	$(MAKE) $(MODEL_INVERSION48_TARGETS)

experiment-inversions-all: experiment-inversion-all experiment-inversion48-all

experiment-clip-all:
	$(MAKE) $(MODEL_CLIP_TARGETS) SKIP_MODEL_DOWNLOAD=1

experiment-clip48-all:
	$(MAKE) $(MODEL_CLIP48_TARGETS) SKIP_MODEL_DOWNLOAD=1

experiment-clips-all: experiment-clip-all experiment-clip48-all

smoke:
	$(PYTHON) -m compileall -q src

clean:
	rm -rf build dist *.egg-info
	rm -rf outputs
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
