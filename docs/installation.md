# Installation

The code needs Python 3.10 to 3.12 and PyTorch 2.1 or newer.

A CUDA GPU is recommended:

| Hardware and setting | Time per tree |
|---|---|
| RTX 4090, full model (both resolutions, 5 folds, 20 augmented passes) | about 2 s, after a one-time load of about 30 s |
| CPU (16 threads), full model | about 4 min |
| CPU (16 threads), one fold without augmentation (`--folds 0 --tta 1`) | about 3 s |

## uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the exact versions in the committed `uv.lock`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is not installed
uv sync                                                # inference and evaluation
uv sync --extra geometry --extra train --extra dev     # + LightGBM branch, training extras, tests (= make install)
```

On Linux and Windows this installs the PyTorch CUDA 12.8 wheels, which need NVIDIA driver 570 or newer. For an older
driver, change the index URL in `pyproject.toml` (`[[tool.uv.index]]`) to `.../whl/cu126` or `.../whl/cu124`.
`predict.sh` picks up `.venv` automatically.

## pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA or CPU build
pip install -e ".[geometry,train,dev]"
```

To install the exact versions the release was tested with instead:

```bash
pip install -r requirements/train.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

`requirements/inference.txt` holds the inference-only subset.

## conda

```bash
conda env create -f requirements/environment.yml
conda activate mvtree
```

## Docker

```bash
docker build -f docker/Dockerfile -t three-lidars .
./predict.sh --docker examples/05554_Pinus_sylvestris_ULS.laz      # mounts ./weights and the input folder
```

or without `predict.sh`:

```bash
docker run --rm --gpus all -v $PWD/weights:/app/weights -v $PWD/examples:/data \
    three-lidars predict /data/05554_Pinus_sylvestris_ULS.laz
```

The image bundles the CUDA runtime through the PyTorch wheels. The host needs an NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
Drop `--gpus all` to run on CPU. To run the web app in Docker, see [Web app](web_app.md#docker).

## Check the setup

```bash
python -m mvtree info      # versions, GPU, installed weights
make test                  # unit tests (CPU; the prediction test also runs when weights are installed)
```

Next: [download the weights](../weights/README.md) and [run a prediction](inference.md).
