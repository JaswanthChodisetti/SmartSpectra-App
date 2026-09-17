"""
Inference vs ground-truth comparison for the SmartSpectra demo.

Takes an RGB photo, runs the fine-tuned MST++ checkpoint, loads the
matching .mat hyperspectral cube, and reports per-band + overall
MRAE, RMSE, PSNR. Saves a side-by-side band montage to disk so you
can drop it into the report.

Usage:
    python demo/compare_inference_vs_gt.py \
        --rgb data/raw/Test_RGB/Test_100.jpg \
        --gt  data/raw/Test_Spec/Test_100.mat \
        --out demo/hsi_samples/Test_100_comparison.png \
        --bands 2 7 15 22 28

Band numbers are 1-indexed and pick representative wavelengths across
the 400-1000 nm range (e.g. 2=440nm, 7=540nm, 15=700nm, 22=840nm,
28=940nm for Agro-HSR's 31 bands).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec


# Reuse the demo's inference path resolver and model loader.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from inference import load_model, run_model1  # noqa: E402


# Agro-HSR 31-band wavelengths (nm), from 400 to 1000 in 20 nm steps.
BAND_WAVELENGTHS_NM = np.arange(400, 1001, 20)
assert len(BAND_WAVELENGTHS_NM) == 31


def load_ground_truth(mat_path: Path) -> np.ndarray:
    """Load Agro-HSR .mat ground truth, return (H, W, 31) float32."""
    m = hdf5storage.loadmat(str(mat_path))
    if "cube" not in m:
        raise KeyError(
            f"Expected key 'cube' in {mat_path}, got {list(m.keys())}"
        )
    cube = np.asarray(m["cube"], dtype=np.float32)
    if cube.shape[-1] != 31:
        raise ValueError(
            f"Expected 31 bands, got shape {cube.shape} from {mat_path}"
        )
    return cube


def align_spatial(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop or pad the pred to the GT's spatial size.

    MST++'s forward uses reflect padding to multiples of 8, so output
    spatial size matches input. If RGB and GT differ in H/W, center-crop
    or zero-pad the prediction to match the GT.
    """
    if pred.shape[:2] == gt.shape[:2]:
        return pred, gt
    ph, pw = pred.shape[:2]
    gh, gw = gt.shape[:2]
    # Center-crop or zero-pad pred to GT size.
    out = np.zeros((gh, gw, pred.shape[2]), dtype=pred.dtype)
    h = min(ph, gh)
    w = min(pw, gw)
    sy_p = (ph - h) // 2
    sx_p = (pw - w) // 2
    sy_g = (gh - h) // 2
    sx_g = (gw - w) // 2
    out[sy_g:sy_g + h, sx_g:sx_g + w] = pred[sy_p:sy_p + h, sx_p:sx_p + w]
    return out, gt


def per_band_metrics(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MRAE, RMSE, PSNR per band. Returns three (31,) arrays.

    MRAE = mean(|pred - gt| / (|gt| + eps))
    RMSE = sqrt(mean((pred - gt)^2))
    PSNR = 10 * log10(1 / MSE), clipped to a sane upper bound.
    """
    eps = 1e-3
    abs_err = np.abs(pred - gt)
    mrae = abs_err.mean(axis=(0, 1)) / (np.abs(gt).mean(axis=(0, 1)) + eps)
    rmse = np.sqrt(((pred - gt) ** 2).mean(axis=(0, 1)))
    mse = ((pred - gt) ** 2).mean(axis=(0, 1))
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
    return mrae.astype(np.float32), rmse.astype(np.float32), psnr.astype(np.float32)


def save_band_montage(
    rgb_path: Path,
    pred: np.ndarray,
    gt: np.ndarray,
    band_indices: list[int],
    out_path: Path,
) -> None:
    """Save a 4-row montage: RGB | predicted | ground truth | |error|.

    One column per selected band. Each band's three panels share the
    same grayscale stretch so visual comparison is fair.
    """
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != gt.shape[:2]:
        rgb = cv2.resize(rgb, (gt.shape[1], gt.shape[0]))

    n = len(band_indices)
    fig = plt.figure(figsize=(3 * n, 11))
    gs = gridspec.GridSpec(4, n, hspace=0.25, wspace=0.05)

    for col, b in enumerate(band_indices):
        wl = BAND_WAVELENGTHS_NM[b]
        g = gt[:, :, b]
        p = pred[:, :, b]
        e = np.abs(g - p)
        # Per-band stretch: [2nd, 98th] percentile of GT, applied to both.
        lo, hi = np.percentile(g, [2, 98])
        hi = max(hi, lo + 1e-6)

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(rgb)
        ax.set_title(f"RGB", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(np.clip((p - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Pred band {b + 1}\n({wl} nm)", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[2, col])
        ax.imshow(np.clip((g - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"GT band {b + 1}\n({wl} nm)", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[3, col])
        # Error: use a viridis-like stretch on the raw error.
        ax.imshow(e, cmap="magma")
        ax.set_title(f"|GT − Pred|", fontsize=9)
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb", required=True, help="Path to RGB jpg/png.")
    p.add_argument("--gt", required=True, help="Path to ground-truth .mat cube.")
    p.add_argument("--ckpt", default=None, help="Path to .pth (default: demo's resolver).")
    p.add_argument("--out", required=True, help="Output montage PNG.")
    p.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=[1, 6, 14, 21, 27],
        help="1-indexed bands to show in the montage (default: 1 6 14 21 27).",
    )
    args = p.parse_args()

    rgb_path = Path(args.rgb)
    gt_path = Path(args.gt)
    out_path = Path(args.out)

    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare] device: {device}")
    model, device = load_model(args.ckpt, device)
    print(f"[compare] RGB: {rgb_path}")
    print(f"[compare] GT:  {gt_path}")

    pred = run_model1(rgb_path, model, device)         # (H, W, 31)
    gt = load_ground_truth(gt_path)                    # (H, W, 31)
    pred, gt = align_spatial(pred, gt)

    print(
        f"[compare] pred  shape={pred.shape} "
        f"min={pred.min():.4f} max={pred.max():.4f} mean={pred.mean():.4f}"
    )
    print(
        f"[compare] gt    shape={gt.shape}   "
        f"min={gt.min():.4f}  max={gt.max():.4f}  mean={gt.mean():.4f}"
    )

    mrae, rmse, psnr = per_band_metrics(pred, gt)
    print("\n[compare] Overall metrics:")
    print(f"  MRAE = {mrae.mean():.4f}")
    print(f"  RMSE = {rmse.mean():.4f}")
    print(f"  PSNR = {psnr.mean():.2f} dB")

    print("\n[compare] Per-band (band, wavelength_nm, MRAE, RMSE, PSNR):")
    for b in range(31):
        print(
            f"  {b + 1:2d}  {BAND_WAVELENGTHS_NM[b]:4d} nm  "
            f"MRAE={mrae[b]:.4f}  RMSE={rmse[b]:.4f}  PSNR={psnr[b]:.2f} dB"
        )

    band_indices_0based = [b - 1 for b in args.bands]
    save_band_montage(rgb_path, pred, gt, band_indices_0based, out_path)
    print(f"\n[compare] Saved montage → {out_path}")

    # Also dump a tiny JSON for the report.
    import json
    metrics_json = out_path.with_suffix(".json")
    metrics_json.write_text(
        json.dumps(
            {
                "rgb": str(rgb_path),
                "gt": str(gt_path),
                "ckpt": str(args.ckpt) if args.ckpt else "<demo default>",
                "overall": {
                    "MRAE": float(mrae.mean()),
                    "RMSE": float(rmse.mean()),
                    "PSNR_dB": float(psnr.mean()),
                },
                "per_band": [
                    {
                        "band": b + 1,
                        "wavelength_nm": int(BAND_WAVELENGTHS_NM[b]),
                        "MRAE": float(mrae[b]),
                        "RMSE": float(rmse[b]),
                        "PSNR_dB": float(psnr[b]),
                    }
                    for b in range(31)
                ],
            },
            indent=2,
        )
    )
    print(f"[compare] Saved metrics JSON → {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
