#!/usr/bin/env python3
"""Train one cross-validation fold of the multi-view classifier.

    python scripts/train/train.py --config configs/convnextv2_256.yaml --fold 0 --out runs/convnextv2_256
    python scripts/train/train.py --config configs/convnextv2_256.yaml --fold 0 \
        --split splits/source_holdout_5fold.csv --out runs/convnextv2_256_source_holdout

Recipe (paper Sec. 5 and Supp. Table A1): three ConvNeXtV2-Base backbones,
logit-adjusted cross-entropy with the fold's own class prior, AdamW, 3 epochs
head-only warmup then 37 epochs of full fine-tuning under a cosine schedule
(backbone LR 3e-5, head LR 3e-4), effective batch 32, mixed precision, gradient
clipping at 1.0, and the checkpoint with the best validation macro-F1 is kept.

Outputs in ``--out``:
    best_fold{k}.pt      model weights of the best epoch
    fold{k}_last.pt      latest epoch, for --resume
    oof_fold{k}.npz      validation probabilities of the best epoch (treeID, fold, probs)
    fold{k}_history.csv  per-epoch metrics
Export the best checkpoints for the predictor with scripts/export/export_weights.py.
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.constants import SPECIES  # noqa: E402
from mvtree.data import TreeRenderDataset, has_renders  # noqa: E402
from mvtree.losses import LogitAdjustedCrossEntropy, class_prior_from_labels  # noqa: E402
from mvtree.model import MultiViewTreeClassifier  # noqa: E402
from mvtree.splits import load_split  # noqa: E402

log = logging.getLogger("train")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", help="override the config's split CSV")
    ap.add_argument("--renders", help="override the config's render directory")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--accum-steps", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--seed", type=int, default=None, help="seed torch/numpy (the paper runs were unseeded)")
    ap.add_argument("--resume", action="store_true", help="continue from fold{k}_last.pt")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train and N val trees")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("split", "renders", "epochs", "batch_size", "accum_steps", "num_workers"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    return args, cfg


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for side, topbot, detail, height, y in loader:
        side, topbot, detail, height, y = (t.to(device, non_blocking=True) for t in (side, topbot, detail, height, y))
        with torch.autocast(device_type=device.split(":")[0], enabled=amp):
            logits = model(side, topbot, detail, height)
            loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(logits.float().softmax(-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    pred = probs.argmax(1)
    return (loss_sum / max(n, 1), accuracy_score(labels, pred),
            f1_score(labels, pred, average="macro", zero_division=0), probs)


def make_phase(model, cfg, phase, epochs):
    """Phase 1: head only. Phase 2: everything, with differential learning rates."""
    if phase == 1:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier")
        opt = AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=cfg["freeze_epochs"], eta_min=cfg["lr_head"] * 0.01)
    else:
        for p in model.parameters():
            p.requires_grad = True
        opt = AdamW([{"params": model.backbone_parameters(), "lr": cfg["lr_backbone"]},
                     {"params": model.head_parameters(), "lr": cfg["lr_head"]}],
                    weight_decay=cfg["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=epochs - cfg["freeze_epochs"], eta_min=cfg["lr_backbone"] * 0.01)
    return opt, sched


def main():
    args, cfg = parse_args()
    os.makedirs(args.out, exist_ok=True)
    k = args.fold
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.out, f"fold{k}.log")),
                                  logging.StreamHandler()])
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.get("amp", True)) and device == "cuda"
    epochs = int(cfg["epochs"])
    log.info(f"fold {k} | config {args.config} | {cfg}")

    meta = pd.read_csv(cfg["metadata"])
    meta["label"] = meta.species.map({s: i for i, s in enumerate(SPECIES)})
    meta = meta[[has_renders(cfg["renders"], t) for t in meta.treeID]].reset_index(drop=True)
    train_ids, val_ids = load_split(cfg["split"], k)
    # Row order = metadata order, so OOF rows line up with the released results.
    train_meta = meta[meta.treeID.isin(train_ids)].reset_index(drop=True)
    val_meta = meta[meta.treeID.isin(val_ids)].reset_index(drop=True)
    if args.limit:
        train_meta = train_meta.sample(min(args.limit, len(train_meta)), random_state=0).sort_index()
        val_meta = val_meta.sample(min(args.limit, len(val_meta)), random_state=0).sort_index()
    log.info(f"trees with renders: {len(meta)} | train {len(train_meta)} | val {len(val_meta)}")

    def dataset(m, augment):
        return TreeRenderDataset(m.treeID.values, m.label.values, m.tree_H.values, cfg["renders"],
                                 img_size=cfg["img_size"], augment=augment)

    # No class-balanced sampler: logit adjustment already corrects for the prior.
    train_loader = DataLoader(dataset(train_meta, True), batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset(val_meta, False), batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    model = MultiViewTreeClassifier(len(SPECIES), backbone=cfg["backbone"], dropout=cfg["dropout"],
                                    pretrained=True, img_size=cfg.get("backbone_img_size")).to(device)
    log.info(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    prior = class_prior_from_labels(train_meta.label.values, len(SPECIES))
    criterion = LogitAdjustedCrossEntropy(prior, tau=cfg["logit_adjust_tau"],
                                          label_smoothing=cfg["label_smoothing"]).to(device)
    scaler = torch.amp.GradScaler(enabled=amp)

    start, best_f1, phase = 1, -1.0, 0
    opt = sched = None
    last_path = os.path.join(args.out, f"fold{k}_last.pt")
    if args.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start, best_f1, phase = ck["epoch"] + 1, ck["best_f1"], ck["phase"]
        opt, sched = make_phase(model, cfg, phase, epochs)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        log.info(f"resumed at epoch {start} (phase {phase}, best macro-F1 {best_f1:.4f})")

    if args.wandb:
        import wandb
        wandb.init(project="three-lidars", name=f"{os.path.basename(args.out)}_fold{k}", config=cfg)

    hist_path = os.path.join(args.out, f"fold{k}_history.csv")
    accum = int(cfg["accum_steps"])
    for epoch in range(start, epochs + 1):
        want = 1 if epoch <= cfg["freeze_epochs"] else 2
        if want != phase:
            phase = want
            opt, sched = make_phase(model, cfg, phase, epochs)
            log.info(f"phase {phase}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")
        t0 = time.time()
        model.train()
        opt.zero_grad()
        n_batches, loss_sum, correct, seen = len(train_loader), 0.0, 0, 0
        for i, (side, topbot, detail, height, y) in enumerate(train_loader):
            side, topbot, detail, height, y = (t.to(device, non_blocking=True)
                                               for t in (side, topbot, detail, height, y))
            with torch.autocast(device_type=device, enabled=amp):
                logits = model(side, topbot, detail, height)
                loss = criterion(logits, y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            loss_sum += loss.item() * accum * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        sched.step()

        val_loss, val_oa, val_f1, probs = evaluate(model, val_loader, criterion, device, amp)
        row = {"epoch": epoch, "phase": phase, "train_loss": loss_sum / max(seen, 1),
               "train_oa": correct / max(seen, 1), "val_loss": val_loss, "val_oa": val_oa,
               "val_macro_f1": val_f1, "lr_head": opt.param_groups[-1]["lr"], "seconds": time.time() - t0}
        log.info(" | ".join(f"{a}={b:.2e}" if a.startswith("lr") else f"{a}={b:.4f}" if isinstance(b, float)
                            else f"{a}={b}" for a, b in row.items()))
        with open(hist_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if fh.tell() == 0:
                w.writeheader()
            w.writerow(row)
        if args.wandb:
            wandb.log(row, step=epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_oa": val_oa,
                        "val_macro_f1": val_f1, "backbone": cfg["backbone"], "img_size": cfg["img_size"],
                        "species": SPECIES}, os.path.join(args.out, f"best_fold{k}.pt"))
            np.savez_compressed(os.path.join(args.out, f"oof_fold{k}.npz"), treeID=val_meta.treeID.values,
                                fold=np.full(len(val_meta), k), probs=probs.astype(np.float32),
                                species=np.array(SPECIES))
            log.info(f"  new best: macro-F1 {val_f1:.4f}, OA {val_oa:.4f}")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "phase": phase, "best_f1": best_f1}, last_path)

    log.info(f"done. best validation macro-F1 {best_f1:.4f}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
