"""
Apple-domain fine-tune of MST++ on synthesized (rgb, hsi_31band) pairs.

Works for two data sources (select via --data_root):
  - Vaishnavi / Puvi Lakshmi (24 pairs, has fresh/fungicide/insecticide
    label field — used by run_apple_finetuned checkpoint).
  - SpectroFood (240 pairs, no label field — all apple reflectance,
    used by run_apple_spectrofood_finetuned checkpoint).

Pipeline:
  1. Load <data_root>/manifest.json
  2. Custom Dataset that resizes (rgb, hsi_31band) to 128x128 patches
  3. Continue training from models/checkpoints/run_trained/net_best.pth
     with low LR (3e-5), short run (5 epochs by default; SpectroFood
     uses 3 since we have 10x more data)
  4. Save checkpoint to --out_dir/net_best.pth

The training script is intentionally minimal: it doesn't use the full MST++
multi-scale training pipeline (which assumes 1000+ images). For small
datasets we use simple random crops with no rotation/flip augmentation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path("/mnt/c/Users/jaswa/ollama/Smart_spectra")
sys.path.insert(0, str(ROOT / "code/MST-plus-plus/train_code/architecture"))
from MST_Plus_Plus import MST_Plus_Plus  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_PAIRS_ROOT = ROOT / "data/pesticides/vaishnavi_2023/apple_pairs"
DEFAULT_CKPT_IN = ROOT / "models/checkpoints/run_trained/net_best.pth"
DEFAULT_CKPT_OUT_DIR = ROOT / "models/checkpoints/run_apple_finetuned"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CROP = 128           # patch size
BATCH = 4            # small batch (24 images total)
INIT_LR = 3e-5       # low LR — gentle continuation fine-tune
DEFAULT_EPOCHS = 5
STRIDE = 64          # 64-pixel stride = ~6 patches per cube at 128 crop
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ApplePairDataset(Dataset):
    """Loads (rgb.png, hsi_31band.npy) pairs and yields random 128x128 crops
    with stride-based indexing. Both rgb and hsi are resized to a common
    spatial size (resizing 31-band HSI preserves per-pixel spectrum).

    Optional per-pair 'label' field is supported (Vaishnavi has
    fresh/fungicide/insecticide; SpectroFood has none). When missing,
    label is recorded as 0 — the training loop here ignores labels
    anyway, so this is purely informational.
    """

    def __init__(self, manifest_path: Path, crop: int = 128, stride: int = 64):
        m = json.loads(manifest_path.read_text())
        self.pairs = list(m["pairs"].values())
        self.crop = crop
        self.stride = stride

        # Pre-load and resize all pairs once (small dataset).
        # Use common spatial target = min(H, W) crop region to avoid padding.
        self.rgb_data: list[np.ndarray] = []  # (3, H, W) float32 in [0,1]
        self.hsi_data: list[np.ndarray] = []  # (31, H, W) float32 in [0,1]
        self.patch_offsets: list[tuple[int, int]] = []  # (img_idx, y, x)
        self.label_idx: list[int] = []  # 0=fresh, 1=fungicide, 2=insecticide; default 0

        cls_to_idx = {"fresh": 0, "fungicide": 1, "insecticide": 2}

        for p in self.pairs:
            # RGB (uint8) -> float32 [0, 1] (per-image min-max matches training)
            rgb = np.array(Image.open(p["rgb_path"])).astype(np.float32)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = rgb.transpose(2, 0, 1)  # (3, H, W)

            # HSI (31, H, W) float32
            hsi = np.load(p["hsi_path"]).astype(np.float32)

            # Resize both to a common size (smallest dims across all pairs)
            # Use 256x256 — fast, and at 128-crop gives ~4 patches per side.
            from scipy.interpolate import interp2d  # not ideal; use np indexing instead
            # We don't have PIL resize for 31 channels. Use torch interpolate.
            t_hsi = torch.from_numpy(hsi).unsqueeze(0)  # (1, 31, H, W)
            t_rgb = torch.from_numpy(rgb).unsqueeze(0)  # (1, 3, H, W)
            target = (256, 256)
            t_hsi = torch.nn.functional.interpolate(
                t_hsi, size=target, mode="bilinear", align_corners=False,
            ).squeeze(0).numpy()
            t_rgb = torch.nn.functional.interpolate(
                t_rgb, size=target, mode="bilinear", align_corners=False,
            ).squeeze(0).numpy()
            self.hsi_data.append(t_hsi)
            self.rgb_data.append(t_rgb)
            # Optional label — defaults to 0 if missing (e.g. SpectroFood).
            self.label_idx.append(cls_to_idx.get(p.get("label", "fresh"), 0))

            # Patch grid
            H, W = t_hsi.shape[1], t_hsi.shape[2]
            for y in range(0, H - crop + 1, stride):
                for x in range(0, W - crop + 1, stride):
                    self.patch_offsets.append((len(self.rgb_data) - 1, y, x))

        print(f"ApplePairDataset: {len(self.rgb_data)} pairs, "
              f"{len(self.patch_offsets)} patches, "
              f"class counts: {np.bincount(self.label_idx, minlength=3).tolist()}")

    def __len__(self) -> int:
        return len(self.patch_offsets)

    def __getitem__(self, idx: int):
        img_idx, y, x = self.patch_offsets[idx]
        rgb = self.rgb_data[img_idx][:, y:y+self.crop, x:x+self.crop]
        hsi = self.hsi_data[img_idx][:, y:y+self.crop, x:x+self.crop]
        return torch.from_numpy(rgb), torch.from_numpy(hsi)


def load_state_dict_flexible(path: Path) -> dict:
    """Same logic as scripts/infer_mst.py._load_state_dict_flexible."""
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint type {type(obj).__name__}")
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
    if tensor_keys:
        return {k: obj[k] for k in tensor_keys}
    raise ValueError(f"No state_dict in checkpoint: {list(obj.keys())[:10]}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Apple-domain fine-tune of MST++ on (rgb, hsi_31band) pairs."
    )
    p.add_argument(
        "--data_root", type=Path, default=DEFAULT_PAIRS_ROOT,
        help="Path to a directory containing manifest.json with pairs",
    )
    p.add_argument(
        "--ckpt_in", type=Path, default=DEFAULT_CKPT_IN,
        help="Source checkpoint to continue from",
    )
    p.add_argument(
        "--out_dir", type=Path, default=DEFAULT_CKPT_OUT_DIR,
        help="Output directory; will hold net_best.pth + training.log",
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=INIT_LR)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--data_label", default="apple pairs",
                   help="Human-readable description of the training data, "
                        "recorded in the saved checkpoint.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = args.out_dir / "net_best.pth"

    print(f"=== Apple-domain fine-tune of MST++ ===")
    print(f"  Init LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"  Crop: {CROP}x{CROP}, Stride: {STRIDE}")
    print(f"  Device: {DEVICE}")
    print(f"  Source ckpt: {args.ckpt_in}")
    print(f"  Output dir: {args.out_dir}\n")

    # Build model + load checkpoint
    model = MST_Plus_Plus().to(DEVICE)
    sd = load_state_dict_flexible(args.ckpt_in)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    # Build dataset
    ds = ApplePairDataset(args.data_root / "manifest.json", crop=CROP, stride=STRIDE)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    # Optimizer + loss
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    # Train
    model.train()
    best_loss = float("inf")
    log = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for rgb, hsi in dl:
            rgb = rgb.to(DEVICE)
            hsi = hsi.to(DEVICE)
            pred = model(rgb)
            loss = loss_fn(pred, hsi)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0
        log.append({"epoch": epoch, "loss": avg, "secs": dt})
        print(f"Epoch {epoch}/{args.epochs}: loss={avg:.6f}  ({dt:.1f}s)", flush=True)
        if avg < best_loss:
            best_loss = avg
            # Save best
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg,
                    "source_ckpt": str(args.ckpt_in),
                    "training_data": args.data_label,
                },
                str(ckpt_out),
            )
            print(f"  -> saved best ckpt to {ckpt_out.name}")

    # Save training log
    (args.out_dir / "training.log").write_text(json.dumps({
        "epochs": args.epochs, "init_lr": args.lr, "batch": args.batch,
        "crop": CROP, "stride": STRIDE, "device": str(DEVICE),
        "data_root": str(args.data_root),
        "training_data": args.data_label,
        "best_loss": best_loss, "log": log,
    }, indent=2))
    print(f"\nDone. Best L1 loss: {best_loss:.6f}")
    print(f"  Checkpoint: {ckpt_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())