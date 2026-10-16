#!/usr/bin/env python3
"""Train one cross-validation fold on a Cloud TPU (PyTorch/XLA).

    python tpu/train_tpu.py --config configs/convnextv2_256.yaml --fold 0 --out runs/tpu_convnextv2_256
    python tpu/train_tpu.py --config configs/convnextv2_256.yaml --fold 0 --out runs/tpu_convnextv2_256 --resume

The recipe is that of scripts/train/train.py (paper Sec. 5, Supp. Table A1):
logit-adjusted cross-entropy with the fold's class prior, AdamW, 3 head-only
epochs then full fine-tuning under a cosine schedule (backbone LR 3e-5, head LR
3e-4), effective batch 32, gradient clipping at 1.0, and the checkpoint with
the best validation macro-F1 is kept. What changes on a TPU:

- one process per chip, data-parallel: the replicas start from the master's
  weights, and gradients are averaged over the chips before clipping, so every
  optimiser step still sees 32 trees;
- AdamW and clipping run on parameters stacked by shape (tpu/stacked_adamw.py),
  with the same arithmetic; per-tensor updates made each step 5x slower;
- the effective batch (batch_size x accum_steps in the config) is split into
  chips x per-chip batch x accumulation steps, e.g. 4 x 8 x 1 on a v6e-4 at 256 px;
- bfloat16 autocast instead of fp16 with a GradScaler;
- validation is sharded over the chips and gathered back in metadata order.

The outputs in --out have the names and keys of scripts/train/train.py
(best_fold{k}.pt, fold{k}_last.pt, oof_fold{k}.npz, fold{k}_history.csv), so
scripts/eval/evaluate_oof.py and scripts/export/export_weights.py read them
unchanged.
"""

import argparse
import csv
import importlib.util
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xla_common  # noqa: E402  (puts the repository root on sys.path)
from stacked_adamw import StackedAdamW  # noqa: E402

from mvtree.constants import SPECIES  # noqa: E402
from mvtree.data import TreeRenderDataset, has_renders  # noqa: E402
from mvtree.losses import LogitAdjustedCrossEntropy, class_prior_from_labels  # noqa: E402
from mvtree.model import MultiViewTreeClassifier  # noqa: E402
from mvtree.splits import load_split  # noqa: E402

log = logging.getLogger("train_tpu")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", help="override the config's split CSV")
    ap.add_argument("--renders", help="override the config's render directory")
    ap.add_argument("--metadata", help="override the config's metadata CSV")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--per-chip-batch", type=int,
                    help="trees per chip per step (default: up to 8 at 256 px, 2 at 512 px)")
    ap.add_argument("--num-workers", type=int, help="data-loader workers per chip (default: from the CPU count)")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--chips", default="all", choices=["all", "1"], help="TPU chips to use on this host")
    ap.add_argument("--seed", type=int, default=None, help="seed torch/numpy (the paper runs were unseeded)")
    ap.add_argument("--resume", action="store_true", help="continue from fold{k}_last.pt")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train and N val trees")
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("split", "renders", "metadata", "epochs"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    return args, cfg


def gpu_recipe():
    """scripts/train/train.py as a module, to share its optimiser phases (parameter
    groups, learning rates, schedule), with StackedAdamW in place of AdamW."""
    path = os.path.join(xla_common.REPO, "scripts", "train", "train.py")
    spec = importlib.util.spec_from_file_location("mvtree_gpu_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AdamW = StackedAdamW
    return module


def batch_plan(cfg, world: int, per_chip=None):
    """Split the config's effective batch into (per-chip batch, accumulation steps)."""
    total = int(cfg["batch_size"]) * int(cfg["accum_steps"])
    if per_chip is None:
        per_chip = max(1, min(total // world, round(8 * (256 / int(cfg["img_size"])) ** 2)))
    if total % (world * per_chip):
        raise SystemExit(f"effective batch {total} does not split into {world} chips x {per_chip} trees; "
                         f"set --per-chip-batch")
    return per_chip, total // (world * per_chip)


def load_fold_data(cfg, fold: int, limit: int = 0):
    meta = pd.read_csv(cfg["metadata"])
    meta["label"] = meta.species.map({s: i for i, s in enumerate(SPECIES)})
    meta = meta[[has_renders(cfg["renders"], t) for t in meta.treeID]].reset_index(drop=True)
    train_ids, val_ids = load_split(cfg["split"], fold)
    # Row order = metadata order, so OOF rows line up with the released results.
    train_meta = meta[meta.treeID.isin(train_ids)].reset_index(drop=True)
    val_meta = meta[meta.treeID.isin(val_ids)].reset_index(drop=True)
    if limit:
        train_meta = train_meta.sample(min(limit, len(train_meta)), random_state=0).sort_index()
        val_meta = val_meta.sample(min(limit, len(val_meta)), random_state=0).sort_index()
    return meta, train_meta.reset_index(drop=True), val_meta.reset_index(drop=True)


class ValShard(Dataset):
    """One chip's share of the validation set, padded to whole batches.

    Items carry their row in the validation set (-1 for padding), so the
    gathered logits can be put back in metadata order.
    """

    def __init__(self, dataset, n_rows: int, world: int, rank: int, batch: int) -> None:
        per_chip = math.ceil(n_rows / (world * batch)) * batch
        rows = list(range(rank, n_rows, world))
        self.dataset = dataset
        self.rows = rows + [-1] * (per_chip - len(rows))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return (*self.dataset[max(row, 0)], row)


@torch.no_grad()
def evaluate(model, loader, labels, criterion, precision):
    """Validation loss, OA, macro-F1, and (n, 33) probabilities, on every chip."""
    import torch_xla.core.xla_model as xm
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    chunks = []
    for side, topbot, detail, height, _, row in loader:
        with xla_common.autocast(precision):
            logits = model(side, topbot, detail, height)
        chunks.append(torch.cat([logits.float(), row.float().unsqueeze(1)], dim=1))
    gathered = xm.all_gather(torch.cat(chunks), dim=0).cpu()
    gathered = gathered[gathered[:, -1] >= 0]
    gathered = gathered[torch.argsort(gathered[:, -1])]
    if not torch.equal(gathered[:, -1].long(), torch.arange(len(labels))):
        raise RuntimeError("validation rows were lost or duplicated across chips")
    logits = gathered[:, :-1]
    loss = criterion(logits, torch.as_tensor(labels)).item()
    probs = logits.softmax(-1).numpy()
    pred = probs.argmax(1)
    return (loss, accuracy_score(labels, pred), f1_score(labels, pred, average="macro", zero_division=0), probs)


def save(data, path: str) -> None:
    """Write a checkpoint from the global master; the rename keeps a half-written file from being synced."""
    import torch_xla.core.xla_model as xm

    xm.save(data, path + ".tmp", global_master=True)
    if xm.is_master_ordinal(local=False):
        os.replace(path + ".tmp", path)


def run_chip(index, args, cfg):
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    from torch.utils.data import DataLoader, DistributedSampler

    world, rank = xr.world_size(), xr.global_ordinal()
    xla_common.share_cpu(world)
    master = rank == 0
    device = torch_xla.device()
    k = args.fold
    handlers = [logging.StreamHandler()]
    if master:
        handlers.append(logging.FileHandler(os.path.join(args.out, f"fold{k}.log")))
    logging.basicConfig(level=logging.INFO if master else logging.WARNING,
                        format="%(asctime)s | %(message)s", handlers=handlers, force=True)
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
    recipe = gpu_recipe()
    per_chip, accum = batch_plan(cfg, world, args.per_chip_batch)
    workers = args.num_workers if args.num_workers is not None else \
        max(2, min(24, (os.cpu_count() or 8) // world - 2))
    epochs = int(cfg["epochs"])
    log.info(f"fold {k} | config {args.config} | {cfg}")
    log.info(f"{world} chip(s) x {per_chip} trees x {accum} accumulation steps = effective batch "
             f"{world * per_chip * accum} | {args.precision} | {workers} loader workers per chip")

    meta, train_meta, val_meta = load_fold_data(cfg, k, args.limit)
    log.info(f"trees with renders: {len(meta)} | train {len(train_meta)} | val {len(val_meta)}")

    def dataset(m, augment):
        return TreeRenderDataset(m.treeID.values, m.label.values, m.tree_H.values, cfg["renders"],
                                 img_size=cfg["img_size"], augment=augment)

    loader_opts = {"num_workers": workers, "persistent_workers": workers > 0}
    if workers:
        loader_opts["prefetch_factor"] = 4
    # No class-balanced sampler: logit adjustment already corrects for the prior.
    train_set = dataset(train_meta, True)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True,
                                 seed=args.seed or 0)
    train_loader = DataLoader(train_set, batch_size=per_chip, sampler=sampler, drop_last=True, **loader_opts)
    eval_batch = 2 * per_chip
    val_loader = DataLoader(ValShard(dataset(val_meta, False), len(val_meta), world, rank, eval_batch),
                            batch_size=eval_batch, shuffle=False, **loader_opts)
    train_device = pl.MpDeviceLoader(train_loader, device)
    val_device = pl.MpDeviceLoader(val_loader, device)

    model = MultiViewTreeClassifier(len(SPECIES), backbone=cfg["backbone"], dropout=cfg["dropout"],
                                    pretrained=True, img_size=cfg.get("backbone_img_size"))
    log.info(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    prior = class_prior_from_labels(train_meta.label.values, len(SPECIES))

    def make_criterion():
        return LogitAdjustedCrossEntropy(prior, tau=cfg["logit_adjust_tau"], label_smoothing=cfg["label_smoothing"])

    criterion, val_criterion = make_criterion().to(device), make_criterion()

    start, best_f1, phase = 1, -1.0, 0
    opt = sched = None
    last_path = os.path.join(args.out, f"fold{k}_last.pt")
    ck = None
    if args.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        start, best_f1, phase = ck["epoch"] + 1, ck["best_f1"], ck["phase"]
    model.to(device)
    # Unlike DDP, torch_xla does not synchronise replicas: without this, every chip
    # keeps its own random initialisation of the head and the averaged gradients
    # train four different models.
    xm.broadcast_master_param(model)
    if ck is not None:
        opt, sched = recipe.make_phase(model, cfg, phase, epochs)
        if ck.get("optimizer_class") == type(opt).__name__:
            opt.load_state_dict(ck["optimizer"])
        else:
            log.warning("checkpoint has no StackedAdamW state; resuming with fresh optimiser moments")
        sched.load_state_dict(ck["scheduler"])
        log.info(f"resumed at epoch {start} (phase {phase}, best macro-F1 {best_f1:.4f})")
        del ck

    hist_path = os.path.join(args.out, f"fold{k}_history.csv")
    for epoch in range(start, epochs + 1):
        want = 1 if epoch <= cfg["freeze_epochs"] else 2
        if want != phase:
            phase = want
            opt, sched = recipe.make_phase(model, cfg, phase, epochs)
            log.info(f"phase {phase}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")
        t0 = time.time()
        model.train()
        sampler.set_epoch(epoch)
        opt.zero_grad()
        n_batches = len(train_loader)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i, (side, topbot, detail, height, y) in enumerate(train_device):
            with xla_common.autocast(args.precision):
                logits = model(side, topbot, detail, height)
                loss = criterion(logits, y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                grads = opt.grads()
                if world > 1:
                    xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world)  # mean over chips, before clipping
                opt.clip_(grads, cfg["grad_clip"])
                opt.step(grads)
                opt.zero_grad()
            loss_sum += loss.detach() * accum * y.size(0)
            correct += (logits.argmax(1) == y).sum()
            if master and ((i + 1) in (1, 10) or (i + 1) % 200 == 0):
                rate = (i + 1) * per_chip * world / (time.time() - t0)
                log.info(f"  epoch {epoch} step {i + 1}/{n_batches} | {rate:.1f} trees/s since epoch start "
                         f"| {xla_common.compile_count()} compilations")
        sched.step()
        seen = float(n_batches * per_chip * world)
        totals = xm.all_reduce(xm.REDUCE_SUM, torch.stack([loss_sum, correct])).cpu()
        train_seconds = time.time() - t0

        val_loss, val_oa, val_f1, probs = evaluate(model, val_device, val_meta.label.values, val_criterion,
                                                   args.precision)
        row = {"epoch": epoch, "phase": phase, "train_loss": totals[0].item() / seen,
               "train_oa": totals[1].item() / seen, "val_loss": val_loss, "val_oa": val_oa,
               "val_macro_f1": val_f1, "lr_head": opt.param_groups[-1]["lr"], "seconds": time.time() - t0}
        log.info(" | ".join(f"{a}={b:.2e}" if a.startswith("lr") else f"{a}={b:.4f}" if isinstance(b, float)
                            else f"{a}={b}" for a, b in row.items())
                 + f" | train {seen / train_seconds:.1f} trees/s | {xla_common.compile_count()} compilations")
        if master:
            with open(hist_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if fh.tell() == 0:
                    w.writeheader()
                w.writerow(row)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_oa": val_oa,
                  "val_macro_f1": val_f1, "backbone": cfg["backbone"], "img_size": cfg["img_size"],
                  "species": SPECIES}, os.path.join(args.out, f"best_fold{k}.pt"))
            if master:
                np.savez_compressed(os.path.join(args.out, f"oof_fold{k}.npz"), treeID=val_meta.treeID.values,
                                    fold=np.full(len(val_meta), k), probs=probs.astype(np.float32),
                                    species=np.array(SPECIES))
            log.info(f"  new best: macro-F1 {val_f1:.4f}, OA {val_oa:.4f}")
        save({"model": model.state_dict(), "optimizer": opt.state_dict(), "optimizer_class": type(opt).__name__,
              "scheduler": sched.state_dict(),
              "epoch": epoch, "phase": phase, "best_f1": best_f1}, last_path)

    log.info(f"done. best validation macro-F1 {best_f1:.4f}")


def main(argv=None) -> None:
    args, cfg = parse_args(argv)
    xla_common.require_xla()
    os.makedirs(args.out, exist_ok=True)
    import timm

    timm.create_model(cfg["backbone"], pretrained=True)  # download once, before the chips start
    xla_common.launch(run_chip, (args, cfg), all_chips=args.chips == "all")


if __name__ == "__main__":
    main()
