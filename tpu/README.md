# Running on Cloud TPU

Inference and training on Google Cloud TPUs with [PyTorch/XLA](https://github.com/pytorch/xla). The model, the
released weights, the renders, and the training recipe are the ones in the rest of the repository. Only the device
code is different, so checkpoints move freely between TPU and GPU.

| Script | What it does |
|---|---|
| [`predict_tpu.py`](predict_tpu.py) | classify scans with the full ensemble (both resolutions, 5 folds, 20 TTA) on every chip of the host |
| [`train_tpu.py`](train_tpu.py) | train one cross-validation fold, data-parallel over the chips |
| [`stacked_adamw.py`](stacked_adamw.py) | AdamW and gradient clipping on parameters stacked by shape, which keeps the TPU busy |
| [`train_all_folds_tpu.sh`](train_all_folds_tpu.sh) | train all five folds, evaluate them, and export the weights to `weights_tpu/` |
| [`setup_vm.sh`](setup_vm.sh) | install the environment on a TPU VM ([`requirements.txt`](requirements.txt)) |
| [`cloud/launch.sh`](cloud/launch.sh) | create a TPU VM that runs a whole training job unattended, with everything kept in a GCS bucket |
| [`cloud/watch.sh`](cloud/watch.sh) | follow that job and recreate the VM when a spot TPU is preempted |

Tested with torch 2.9.0 and torch_xla 2.9.0 on a v6e-4 (Trillium) VM, runtime `v2-alpha-tpuv6e`, Python 3.10.

## Setup on a TPU VM

```bash
gcloud compute tpus tpu-vm create my-tpu --zone us-east1-d --accelerator-type v6e-4 --version v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh my-tpu --zone us-east1-d

git clone https://github.com/JaeLee18/FOR20K-lidar-tree-species.git && cd FOR20K-lidar-tree-species
bash tpu/setup_vm.sh                  # .venv-tpu: CPU torch 2.9.0 + torch_xla[tpu] 2.9.0; prints the chips it sees
source .venv-tpu/bin/activate
python -m mvtree download             # released weights into ./weights
```

For v5e use `--version v2-alpha-tpuv5-lite`, for v5p `v2-alpha-tpuv5`, and for v4 `tpu-ubuntu2204-base`.

## Inference

```bash
python tpu/predict_tpu.py examples/                         # every chip, both resolutions, 5 folds, 20 TTA
python tpu/predict_tpu.py scans/ -o predictions.csv         # a folder -> the same CSV as mvtree predict
python tpu/predict_tpu.py tree.laz --chips 1 --folds 0 --tta 1
```

Each chip takes every n-th scan and keeps all ten fold models in its memory (10.6 GB in fp32). CPU worker processes
read and render the next scans while the chip runs the current batch. The output columns, `--resolution`,
`--folds`, `--tta`, `--seed`, and `--weights` are those of `mvtree predict` ([docs/inference.md](../docs/inference.md)).
Options specific to the TPU:

| Option | Default | Meaning |
|---|---|---|
| `--chips` | `all` | `all` chips of the host, or `1` |
| `--trees-per-batch` | 4 | trees per forward pass at 256 px, each with `--tta` copies; a quarter as many at 512 px |
| `--precision` | `bf16` | bf16 autocast, or `fp32` |
| `--render-workers` | auto | CPU processes rendering scans for each chip |

The augmented copies are drawn exactly as in `mvtree.predict`, so the TPU and GPU predictors see the same flips and
rotations. On the four example scans, the TPU's probabilities differ from an RTX 4090's by at most 1.6e-5 with
`--precision fp32` and 1.2e-4 with bf16, and the predicted species are the same.

Measured on a v6e-4, full model (256 and 512 px, 5 folds, 20 augmented passes), bf16:

| Chips | Steady state | Per tree |
|---|---|---|
| 4 | 3.0 trees/s | 0.33 s |
| 1 | 0.75 trees/s | 1.3 s |

An RTX 4090 takes about 2 s per tree. Each run first compiles the networks, which takes about 2 minutes; later
batches reuse the compiled graphs. (torch_xla's persistent compilation cache cannot reload executables on the v6e
runtime, so every run compiles afresh.)

## Training

```bash
make renders RES=256                                        # or restore a render tarball
bash tpu/train_all_folds_tpu.sh configs/convnextv2_256.yaml runs/tpu_convnextv2_256
python tpu/predict_tpu.py examples/ --weights weights_tpu   # predict with the TPU-trained weights
```

`train_tpu.py` follows `scripts/train/train.py` and takes the same configs. It reuses that script's optimiser phases,
so the recipe cannot drift between the two. What changes on a TPU:

- **Data parallel.** There is one process per chip. The replicas start from the master's weights (torch_xla, unlike
  DDP, does not synchronise them), and gradients are averaged across chips before clipping.
- **Same effective batch.** The config's batch (`batch_size x accum_steps` = 32) is split into chips × per-chip
  batch × accumulation steps. A v6e-4 at 256 px runs 4 × 8 × 1; a single chip runs 1 × 8 × 4. Set it with
  `--per-chip-batch`.
- **bfloat16 autocast** replaces fp16 with a GradScaler.
- **The optimiser works on stacks.** PyTorch/XLA traces every operation on the host before the chip runs the
  step. With `torch.optim.AdamW` and `clip_grad_norm_` over the 1,140 parameter tensors of three ConvNeXt
  backbones, that tracing took about 1 s per step, against 75 ms for the chip's work. [`StackedAdamW`](stacked_adamw.py)
  groups the parameters by shape (27 shapes) and runs clipping, the cross-chip average, and the update on each
  stack. The arithmetic matches `torch.optim.AdamW` to 2e-8, and full fine-tuning runs about 6 times faster.
  The values that change during training (learning rates, bias corrections) reach the chip as one small tensor
  per step. As Python floats, torch_xla would cache each as a device scalar by value, and two that round to the
  same float32 (it happens at epochs 6 and 18 of this schedule) would share one. That changes the graph and
  recompiles the whole step, which took 9 minutes at 512 px and once crashed the run.
- **Validation** is sharded over the chips and gathered back in metadata order.

The outputs (`best_fold{k}.pt`, `fold{k}_last.pt`, `oof_fold{k}.npz`, `fold{k}_history.csv`) have the names and keys of
the GPU script. `scripts/eval/evaluate_oof.py` and `scripts/export/export_weights.py` read them unchanged, and
`--resume` continues from the last epoch.

### Results

Both configurations were trained on a v6e-4 with the five random-split folds. Out-of-fold results on all 17,707
labelled trees (`python scripts/eval/evaluate_oof.py`, `--ensemble` for the last rows):

| Model | Trained on | OA | macro-F1 | Balanced acc. | TLS | MLS | ULS | Time per fold |
|---|---|---|---|---|---|---|---|---|
| 256 px | RTX 4090 (paper, released weights) | 0.886 | 0.843 | 0.846 | 0.879 | 0.934 | 0.894 | about 6 h |
| 256 px | TPU v6e-4 | 0.886 | 0.837 | 0.837 | 0.878 | 0.938 | 0.890 | 79 min |
| 512 px | GPU (paper) | 0.894 | 0.855 | | 0.888 | 0.928 | 0.900 | |
| 512 px | H100 (released weights) | 0.895 | 0.856 | 0.854 | 0.890 | 0.932 | 0.895 | about 13 h |
| 512 px | TPU v6e-4 | 0.895 | 0.854 | 0.856 | 0.890 | 0.935 | 0.896 | 3.3 h |
| 256 + 512 | GPU (paper) | 0.899 | 0.860 | | 0.893 | 0.936 | 0.903 | |
| 256 + 512 | GPU (released weights) | 0.899 | 0.861 | 0.856 | 0.894 | 0.938 | 0.899 | |
| 256 + 512 | TPU v6e-4 | 0.899 | 0.854 | 0.851 | 0.894 | 0.939 | 0.902 | |

None of the runs was seeded, so the TPU and GPU runs differ as two GPU runs would. Fold by fold, the TPU's macro-F1
is 0.016 below to 0.007 above the GPU's at 256 px, and 0.008 below to 0.005 above at 512 px. The paired bootstrap
intervals of the differences (256 px: −0.014 to +0.002; 512 px: −0.008 to +0.005) contain zero, and the TPU and GPU
models give the same species for 93% (256 px) and 94% (512 px) of the trees. The two-resolution ensemble has the same
OA and a macro-F1 0.006 lower, mostly from rare species (*Prunus avium*, 42 trees; *Fraxinus excelsior*, 126). With
one run per device, and a bootstrap that resamples the trees but not the training, this cannot be told apart from
run-to-run variation. The TPU-trained weights of both resolutions classify the four example scans correctly. (Folds
0 to 2 at 256 px ran before the recompilation fix above and took 87 minutes each.)

The TPU-trained weights are on Google Drive, in the layout of the released ones (link and checksums in
[`weights_tpu/manifest.json`](../weights_tpu/manifest.json)):

```bash
python -m mvtree download --weights weights_tpu             # into ./weights_tpu, SHA-256 verified
python -m mvtree predict examples/ --weights weights_tpu    # on a GPU or CPU
python tpu/predict_tpu.py examples/ --weights weights_tpu   # on a TPU
```

## Unattended runs on Google Cloud

`cloud/launch.sh` uploads the working tree to a bucket and creates a TPU VM whose boot script
([`cloud/startup.sh`](cloud/startup.sh)) runs the whole job:

1. Install the environment.
2. If the released weights are in `$BUCKET/weights/`, check inference on the TPU: the example scans in bf16 and fp32,
   and a 64-tree throughput run on all chips and on one chip.
3. Restore the renders from `$BUCKET/data/renders_<size>_dev.tar`. The first run instead downloads FOR-species20K
   from Zenodo, renders it with every CPU core, and stores that tarball.
4. Train all five folds, resuming from `$BUCKET/<run>/runs/` and syncing back every 5 minutes.
   If training crashes, it starts again from the last checkpoint, up to three times.
5. Evaluate, export, and predict the examples with the new weights. The VM then writes `DONE` (or `FAILED`) to the
   bucket and deletes itself.

```bash
BUCKET=gs://my-bucket ZONE=europe-west4-a bash tpu/cloud/launch.sh    # v6e-4 spot, 256 px
BUCKET=gs://my-bucket CANDIDATES=$'v6e-4 europe-west4-a v2-alpha-tpuv6e\nv6e-4 us-east1-d v2-alpha-tpuv6e\nv5p-8 us-east5-a v2-alpha-tpuv5' \
  bash tpu/cloud/watch.sh
```

`launch.sh` takes `ZONE`, `TYPE`, `VERSION`, `CONFIG`, `RUN`, `SPOT` (1 = spot, 0 = on-demand), and `AUTO_DELETE` from
the environment. Spot TPUs cost a fraction of on-demand ones but can be preempted, and a zone can run out of
capacity: on the day this was tested, five US zones had no v6e capacity, spot or on-demand, and a spot v6e was
preempted after 7 minutes. `watch.sh` checks every 5 minutes. It looks for the VM in the zones of `CANDIDATES`
(and `ZONE`). When the VM is gone it recreates it from the first `TYPE ZONE VERSION` line that has capacity, and the
job resumes from the last synced epoch. If a zone cannot be listed, it waits rather than start a second VM.

```bash
gcloud storage cat $BUCKET/tpu_convnextv2_256/logs/vm.log | tail          # VM log
gcloud storage cp -r $BUCKET/tpu_convnextv2_256/runs runs/tpu_convnextv2_256
python scripts/eval/evaluate_oof.py runs/tpu_convnextv2_256               # OOF metrics, as in paper Table 2
```

## Acknowledgement

The Cloud TPU compute used to build and test this port was provided by Google through the TPU Builder Award.
