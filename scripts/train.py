import argparse
import time
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config as C
from deeplab_froth.data import FrothLabelMeDataset
from deeplab_froth.models import build_deeplab_model


# --------- utils ---------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if C.device == "cuda":
        return torch.device("cuda")
    if C.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- metrics ---------
@torch.no_grad()
def compute_miou_from_logits(
    logits_or_dict,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    logits = logits_or_dict["out"] if isinstance(logits_or_dict, dict) else logits_or_dict
    pred = logits.argmax(1)   # (B,H,W)
    mask = target != ignore_index

    pred = pred[mask]
    tgt  = target[mask]

    if tgt.numel() == 0:
        return 0.0

    ious = []
    for c in range(num_classes):
        tp = ((pred == c) & (tgt == c)).sum().float()
        fp = ((pred == c) & (tgt != c)).sum().float()
        fn = ((pred != c) & (tgt == c)).sum().float()
        denom = tp + fp + fn
        if denom > 0:
            ious.append((tp / denom).item())

    return sum(ious) / len(ious) if ious else 0.0


# --------- train / val loops ---------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss, n_steps = 0.0, 0

    optimizer.zero_grad(set_to_none=True)
    use_amp = C.use_amp and (device.type == "cuda")
    total_steps = len(loader)
    log_every = max(1, total_steps // 10)  # print ~10 times per epoch

    for step, (imgs, masks) in enumerate(loader, 1):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out    = model(imgs)
            logits = out["out"] if isinstance(out, dict) else out
            loss   = criterion(logits, masks) / C.accum_steps

        scaler.scale(loss).backward()

        if step % C.accum_steps == 0:
            if C.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * C.accum_steps
        n_steps += 1

        # --- progress indicator ---
        if (step % log_every == 0) or (step == 1) or (step == total_steps):
            cur_loss = loss.item() * C.accum_steps
            print(
                f"[DeepLabV3][epoch {epoch:02d}] step {step:04d}/{total_steps:04d} "
                f"batch_loss={cur_loss:.4f}",
                end="\r",
                flush=True,
            )

    print()  # newline after last carriage-return
    return running_loss / max(1, n_steps)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out    = model(imgs)
        logits = out["out"] if isinstance(out, dict) else out
        loss   = criterion(logits, masks).item()
        miou   = compute_miou_from_logits(
            logits, masks, num_classes=C.num_classes, ignore_index=C.ignore_index
        )

        total_loss += loss
        total_iou  += miou
        n += 1

    return total_loss / max(1, n), total_iou / max(1, n)


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLabV3-ResNet101 on froth dataset.")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (default: use Config.epochs).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs is not None:
        C.epochs = args.epochs

    C.setup()
    set_seed(C.seed)
    device = get_device()
    print(f"Using device: {device}")

    # datasets / loaders
    train_ds = FrothLabelMeDataset(C.train_dir, train=True)
    val_ds   = FrothLabelMeDataset(C.val_dir,   train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=C.batch_size_train,
        shuffle=True,
        num_workers=C.num_workers,
        pin_memory=C.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=C.batch_size_val,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=C.pin_memory,
    )

    # model
    model = build_deeplab_model().to(device)

    # loss
    criterion = nn.CrossEntropyLoss(ignore_index=C.ignore_index)

    # optimizer (backbone vs head)
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(),   "lr": C.backbone_lr, "weight_decay": C.weight_decay},
            {"params": model.classifier.parameters(), "lr": C.head_lr,     "weight_decay": C.weight_decay},
        ]
    )

    # poly LR schedule
    def poly(epoch: int):
        return (1 - epoch / max(1, C.epochs)) ** C.power_poly

    scheduler = LambdaLR(optimizer, lr_lambda=[poly, poly])

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and C.use_amp))

    # resume (optional)
    start_epoch = 0
    best_miou, best_epoch = -1.0, -1
    if C.resume_ckpt is not None and Path(C.resume_ckpt).is_file():
        ckpt = torch.load(C.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_miou   = ckpt.get("best_mIoU", best_miou)
        best_epoch  = ckpt.get("best_epoch", best_epoch)
        print(f"Resumed from {C.resume_ckpt}: start={start_epoch}, best_mIoU={best_miou:.4f}@{best_epoch}")

    print(f"Starting training for {C.epochs} epoch(s) with accumulation x{C.accum_steps}…")
    t_global = time.time()

    for epoch in range(start_epoch + 1, C.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        criterion,
        device,
        epoch=epoch,
        )
        val_loss, val_miou = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        # save LAST
        last_path = C.deeplab_out / "last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "num_classes": C.num_classes,
                "classes": C.class_names,
                "best_mIoU": best_miou,
                "best_epoch": best_epoch,
            },
            last_path,
        )

        print(
            f"[DeepLabV3] Epoch {epoch:02d}/{C.epochs} | "
            f"loss={train_loss:.4f} | val_mIoU={val_miou:.4f} | {dt:.1f}s"
        )

        # save BEST
        if val_miou > best_miou:
            best_miou, best_epoch = val_miou, epoch
            best_path = C.deeplab_out / "best_mIoU.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_mIoU": val_miou,
                    "num_classes": C.num_classes,
                    "classes": C.class_names,
                    "best_mIoU": best_miou,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
            print(f"  ✅ New best mIoU {best_miou:.4f} saved:\n"
                  f"     - {best_path}")

    print("\nDone.")
    print(f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"Checkpoints in: {C.deeplab_out}")
    print(f"  last: {C.deeplab_out / 'last.pth'}")
    print(f"  best: {C.deeplab_out / 'best_mIoU.pth'}")
    print(f"Total time: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
