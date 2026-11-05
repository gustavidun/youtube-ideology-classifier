# ==============================================================================
# YouTube Ideology Classifier — Canonical Pipeline Makefile
# ==============================================================================

SHELL := /bin/bash
PYTHON := .venv/bin/python

# Default Configurations & Paths
CONFIG ?= configs/small_ranked.yaml
VIDEOS_PATH ?= data/slants_train.csv
CONTEXT_PATH ?= context.parquet
OUT_PATH ?= pair_similarity.parquet
DECISION_MODEL ?= autotrust/JEV-27B
CONTEXT_MODEL ?= meta-llama/Llama-3.3-70B-Instruct
LOCAL ?= false
OVERWRITE ?= false
.PHONY: help install venv teacher-context teacher-pairs eval-teacher benchmark wrangle train serve-qwen8b serve-jev27b serve-qwen-flash clean all

# ------------------------------------------------------------------------------
# Help / Documentation
# ------------------------------------------------------------------------------
help:
	@echo "=============================================================================="
	@echo " YouTube Ideology Classifier — Pipeline Commands"
	@echo "=============================================================================="
	@echo " Environment:"
	@echo "   make venv              Create virtual environment and install requirements"
	@echo "   make install           Install/update dependencies from requirements.txt"
	@echo ""
	@echo " LLM Teacher Supervision:"
	@echo "   make teacher-context   Run context generation (mode=context)"
	@echo "                          Options: VIDEOS_PATH=... CONTEXT_PATH=... CONTEXT_MODEL=... LOCAL=true/false OVERWRITE=true/false"
	@echo "   make teacher-pairs     Run pairwise scoring (mode=pairs)"
	@echo "                          Options: CONTEXT_PATH=... OUT_PATH=... DECISION_MODEL=... LOCAL=true/false OVERWRITE=true/false"
	@echo "   make eval-teacher      Run teacher evaluation across all conditions on human test set"
	@echo ""
	@echo " Benchmarking & Visualization:"
	@echo "   make benchmark         Evaluate teacher models against human labels and update plots"
	@echo ""
	@echo " Student Model Training:"
	@echo "   make wrangle           Wrangle and split dataset pairs for sentence-transformer"
	@echo "   make train             Train political embedding model using CONFIG (default: $(CONFIG))"
	@echo "                          Options: CONFIG=configs/bge_m3.yaml"
	@echo ""
	@echo " Local Inference Servers (vLLM):"
	@echo "   make serve-qwen8b      Start local Qwen3-8B server (port 8000)"
	@echo "   make serve-jev27b      Start local JEV-27B server (port 8001)"
	@echo "   make serve-qwen-flash  Start local Qwen-Flash-Next server"

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install:
	$(PYTHON) -m pip install -r requirements.txt

# ------------------------------------------------------------------------------
# Teacher LLM Inference Pipeline
# ------------------------------------------------------------------------------
teacher-context:
	$(PYTHON) -m src.teacher \
		--mode context \
		--videos_path $(VIDEOS_PATH) \
		--context_path $(CONTEXT_PATH) \
		--context_model $(CONTEXT_MODEL) \
		$(if $(filter true,$(LOCAL)),--local,--no-local) \
		$(if $(filter true,$(OVERWRITE)),--overwrite,--no-overwrite)

teacher-pairs:
	$(PYTHON) -m src.teacher \
		--mode pairs \
		--context_path $(CONTEXT_PATH) \
		--out_path $(OUT_PATH) \
		--decision_model $(DECISION_MODEL) \
		$(if $(filter true,$(LOCAL)),--local,--no-local) \
		$(if $(filter true,$(OVERWRITE)),--overwrite,--no-overwrite)

eval-teacher:
	$(PYTHON) -m src.eval.teacher

# ------------------------------------------------------------------------------
# Evaluation Benchmark & Leaderboard
# ------------------------------------------------------------------------------
benchmark:
	$(PYTHON) -m src.eval.benchmark

# ------------------------------------------------------------------------------
# Data Wrangling & Student Training
# ------------------------------------------------------------------------------
wrangle:
	$(PYTHON) -m src.wrangle

train:
	$(PYTHON) -m src.model --path $(CONFIG)

all: wrangle train

# ------------------------------------------------------------------------------
# Local vLLM Inference Servers
# ------------------------------------------------------------------------------
serve-qwen8b:
	chmod +x ./serve_qwen8b_spark.sh
	./serve_qwen8b_spark.sh

serve-jev27b:
	chmod +x ./serve_jev27b_spark.sh
	./serve_jev27b_spark.sh

serve-qwen-flash:
	chmod +x ./serve_qwen-flash-next_spark.sh
	./serve_qwen-flash-next_spark.sh

