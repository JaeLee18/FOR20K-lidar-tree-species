# Common tasks. `make help` lists them.
PY ?= python
RES ?= 256

.PHONY: help install weights demo serve test lint data renders splits train-all paper-tables testset docker

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## create .venv with uv (CUDA 12.8 wheels on Linux)
	uv sync --extra geometry --extra train --extra dev

weights:         ## download released weights from Google Drive
	$(PY) -m mvtree download

demo:            ## classify the bundled example scans
	./predict.sh examples/

serve:           ## start the self-hosted web app on http://127.0.0.1:8000
	$(PY) -m mvtree serve

test:            ## unit tests (CPU, no weights needed; prediction test runs if weights exist)
	$(PY) -m pytest -q

lint:            ## ruff
	ruff check mvtree scripts tests tpu

data:            ## download FOR-species20K from Zenodo (~27 GB)
	$(PY) scripts/data/download_data.py

renders:         ## render dev/test at RES px (RES=256 or 512)
	$(PY) scripts/data/render_dataset.py --source data/raw/dev.zip --out data/renders_$(RES)/dev --size $(RES)
	$(PY) scripts/data/render_dataset.py --source data/raw/test.zip --out data/renders_$(RES)/test --size $(RES)

splits:          ## regenerate splits/ (identical to the shipped files)
	$(PY) scripts/data/make_splits.py --out splits

train-all:       ## train 5 folds at RES px and export the weights
	bash scripts/train/train_all_folds.sh configs/convnextv2_$(RES).yaml runs/convnextv2_$(RES)

paper-tables:    ## reproduce the paper's tables from results/oof (CPU, ~1 min)
	bash scripts/eval/reproduce_paper.sh

testset:         ## write a Codabench submission for the hidden test set
	$(PY) scripts/eval/predict_testset.py --source data/raw/test.zip --out submissions/predictions.csv

docker:          ## build the Docker image
	docker build -f docker/Dockerfile -t three-lidars .

