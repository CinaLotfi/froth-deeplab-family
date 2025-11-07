import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader

from config import Config as C
from deeplab_froth.data.froth_dataset import FrothLabelMeDataset
from deeplab_froth.models.deeplabv3 import build_deeplab_model


# --------- utils ---------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if getattr(C, "device", "auto") == "cuda":
        return torch.device("cuda")
    if getattr(C, "device", "auto") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(split: str):
    """
    Build dataset + loader for prediction.

    split: "train" or "val"
    """
    if split == "train":
        root = C.train_dir
        train_flag = True
    else:
        root = C.val_dir
        split = "val"
        train_flag = False

    ds = FrothLabelMeDataset(
        root_dir=root,
        train=train_flag,
    )

    loader = DataLoader(
        ds,
        batch_size=C.batch_size_val,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=(get_device().type == "cuda") and C.pin_memory,
    )

    print(f"Using [{split}] split from: {root} ({len(ds)} samples)")
    return ds, loader



def load_finetuned_deeplab(device: torch.device, ckpt_path: Path | None = None) -> nn.Module:
    """
    Load DeepLabV3-ResNet101 and apply finetuned weights.
    """
    model = build_deeplab_model().to(device)

    if ckpt_path is None:
        ckpt_path = C.deeplab_out / "best_mIoU.pth"
        if not ckpt_path.exists():
            ckpt_path = C.deeplab_out / "last.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"DeepLab checkpoint not found at:\n  {ckpt_path}\n"
            f"Run training first: python -m scripts.train"
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    print(f"Loaded DeepLabV3 finetuned weights from: {ckpt_path}")
    return model


@torch.no_grad()
def predict_split(device: torch.device, split: str, argmax: bool, thr: float):
    ds, loader = build_loader(split)
    model = load_finetuned_deeplab(device)

    # where to save
    model_tag = "deeplabv3_resnet101"
    out_root = C.outputs_root / "pred_masks" / model_tag / split
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving prediction masks to: {out_root}")

    idx_global = 0

    for (imgs, _masks) in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        logits = out["out"] if isinstance(out, dict) else out  # (1, C, H, W)
        probs = torch.softmax(logits, dim=1)                   # (1, C, H, W)

        # binary case: foreground = class 1
        if argmax:
            pred_cls = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # (H,W) in {0,1}
            bin_mask = (pred_cls == 1).astype(np.uint8) * 255
        else:
            fg_prob = probs[0, 1].cpu().numpy()               # (H,W)
            bin_mask = (fg_prob > thr).astype(np.uint8) * 255

        # save
        fname = f"mask_{idx_global:04d}.png"
        out_path = out_root / fname
        cv2.imwrite(str(out_path), bin_mask)
        print(f"Saved: {fname}")
        idx_global += 1

    print("Prediction complete.")


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Predict froth masks with DeepLabV3-ResNet101.")
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to run prediction on.",
    )
    p.add_argument(
        "--argmax",
        action="store_true",
        help="Use argmax over classes instead of probability threshold.",
    )
    p.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Foreground probability threshold if not using argmax.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # wire config
    C.setup()
    set_seed(getattr(C, "seed", 42))
    device = get_device()

    print(f"Using device: {device}")
    predict_split(device, split=args.split, argmax=args.argmax, thr=args.thr)


if __name__ == "__main__":
    main()
