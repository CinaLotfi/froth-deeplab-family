import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config as C
from deeplab_froth.data import FrothLabelMeDataset
from deeplab_froth.models import build_deeplabv3plus_model


# --------- utils ---------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if C.device == "cuda":
        return torch.device("cuda")
    if C.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(split: str) -> DataLoader:
    """
    split: 'train' or 'val'
    """
    if split == "train":
        root = C.train_dir
    else:
        root = C.val_dir

    ds = FrothLabelMeDataset(root, train=False)
    loader = DataLoader(
        ds,
        batch_size=C.batch_size_val,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=C.pin_memory and (get_device().type == "cuda"),
    )
    print(f"Evaluating on {len(ds)} samples from [{split}] dir: {root}")
    return loader


# --------- metrics ---------
@torch.no_grad()
def compute_miou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    pred = logits.argmax(1)
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

        logits = model(imgs)
        loss   = criterion(logits, masks).item()
        miou   = compute_miou_from_logits(
            logits, masks, num_classes=C.num_classes, ignore_index=C.ignore_index
        )

        total_loss += loss
        total_iou  += miou
        n += 1

    return total_loss / max(1, n), total_iou / max(1, n)


# --------- model loading ---------
def load_deeplab_v3plus(device: torch.device, ckpt_path: Path | None = None) -> nn.Module:
    """
    If a finetuned checkpoint exists, load it.
    Otherwise, evaluate ImageNet-pretrained DeepLabV3+.
    """
    if ckpt_path is not None and ckpt_path.exists():
        chosen = ckpt_path
    else:
        best = C.deeplab_v3plus_out / "best_mIoU.pth"
        last = C.deeplab_v3plus_out / "last.pth"
        if best.exists():
            chosen = best
        elif last.exists():
            chosen = last
        else:
            chosen = None

    model = build_deeplabv3plus_model().to(device)

    if chosen is None:
        print(
            f"No DeepLabV3+ checkpoints found in {C.deeplab_v3plus_out}.\n"
            "Evaluating ImageNet-pretrained DeepLabV3+ (ResNet-152, OS=8)."
        )
        model.eval()
        return model

    print(f"Loading DeepLabV3+ checkpoint: {chosen}")
    state = torch.load(chosen, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


# --------- CLI / main ---------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepLabV3+ (ResNet-152) on froth dataset.")
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate on (default: val=test dir).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Optional path to a specific checkpoint (.pth).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    C.setup()
    set_seed(C.seed)
    device = get_device()
    print(f"Using device: {device}")

    loader = build_loader(args.split)
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    model = load_deeplab_v3plus(device, ckpt_path=ckpt_path)

    criterion = nn.CrossEntropyLoss(ignore_index=C.ignore_index)
    val_loss, val_miou = evaluate_one_epoch(model, loader, criterion, device)

    print(f"\n[DeepLabV3+] Eval on split={args.split} → loss={val_loss:.4f}, mIoU={val_miou:.4f}")


if __name__ == "__main__":
    main()
