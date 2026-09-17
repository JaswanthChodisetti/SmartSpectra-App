"""
Extract sample HSI band images from Agro-HSR for faculty demo.

Reads a single .mat (MATLAB v7.3 / HDF5) file and saves:
  - the full 31-band cube as .npy
  - a few single-band PNGs at meaningful wavelengths
    (450 nm blue, 550 nm green, 700 nm red edge, 850 nm NIR)

Output goes to ./hsi_samples/<stem>/ next to this script.

Usage:
    python extract_sample_hsi.py                    # default: Train_1000
    python extract_sample_hsi.py --stem Train_1000
    python extract_sample_hsi.py --stem Test_42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hdf5storage
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 31 bands are evenly spaced from 400 nm to 1000 nm (step = 20 nm).
WAVELENGTHS_NM = np.linspace(400, 1000, 31)
BAND_KEY = "cube"  # top-level key in Agro-HSR .mat files


def _band_index_for(wavelength_nm: float) -> int:
    """Return the integer band index nearest the requested wavelength."""
    return int(np.argmin(np.abs(WAVELENGTHS_NM - wavelength_nm)))


def _normalize_to_uint8(plane: np.ndarray,
                        mode: str = "percentile",
                        percentile_lo: float = 1.0,
                        percentile_hi: float = 99.0) -> np.ndarray:
    """Map a float band plane to a uint8 PNG-friendly image.

    Args:
        plane: 2D float array (one HSI band).
        mode:
          - 'percentile' (default): robust [p_lo, p_hi] stretch → [0, 255].
            Best for typical reflectance data with outliers (e.g. spec edges).
          - 'full': use the plane's true min/max. Preserves relative scale
            across bands but can wash out detail when most pixels are near 0
            (the case for Agro-HSR which has cube mean ~0.04).
          - 'per_band_max': divide each band by its own max. Equivalent to
            'full' if min ≈ 0. Use this when you want each band at the same
            visual brightness regardless of reflectance magnitude.

    Returns:
        uint8 ndarray, same shape as `plane`.
    """
    if mode == "percentile":
        lo, hi = np.percentile(plane, [percentile_lo, percentile_hi])
    elif mode == "full":
        lo, hi = float(plane.min()), float(plane.max())
    elif mode == "per_band_max":
        lo, hi = 0.0, float(plane.max())
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    stretched = (plane - lo) / (hi - lo + 1e-8)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def extract_one(stem: str, data_root: Path, out_root: Path,
                stretch_mode: str = "per_band_max") -> dict:
    """Read one .mat, save cube + band PNGs. Returns a small summary dict."""
    mat_path = data_root / f"{stem}.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(f"No .mat at {mat_path}")

    print(f"[extract] reading {mat_path}")
    mat = hdf5storage.loadmat(str(mat_path))
    if BAND_KEY not in mat:
        raise KeyError(
            f"Expected key '{BAND_KEY}' in {mat_path}, found {list(mat.keys())}"
        )

    cube = mat[BAND_KEY].astype(np.float32)
    print(f"[extract] cube shape={cube.shape}  dtype={cube.dtype}  "
          f"min={cube.min():.4f}  max={cube.max():.4f}  mean={cube.mean():.5f}")

    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save the full cube.
    npy_path = out_dir / f"{stem}_cube.npy"
    np.save(str(npy_path), cube)
    print(f"[extract] saved {npy_path}")

    # 2. Save a few single-band PNGs at visible-to-NIR wavelengths.
    # The .mat returns the cube as (H, W, B) — hdf5storage auto-transposes
    # the MATLAB (B, H, W) layout on read. The band axis is the LAST one.
    band_wavelengths = [450, 550, 700, 850]
    saved_pngs: list[Path] = []
    for wl in band_wavelengths:
        b_idx = _band_index_for(wl)
        actual_wl = WAVELENGTHS_NM[b_idx]
        plane = cube[:, :, b_idx]
        png = _normalize_to_uint8(plane, mode=stretch_mode)
        png_path = out_dir / f"{stem}_band{b_idx:02d}_{int(actual_wl)}nm.png"
        # matplotlib handles uint8 → PNG with no axes, original size.
        plt.imsave(str(png_path), png, cmap="gray", vmin=0, vmax=255)
        saved_pngs.append(png_path)
        print(f"[extract] saved {png_path}")

    # 3. Save a 2x2 montage for at-a-glance faculty review.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, wl, png_path in zip(axes, band_wavelengths, saved_pngs):
        ax.imshow(plt.imread(str(png_path)), cmap="gray")
        b_idx = _band_index_for(wl)
        actual_wl = WAVELENGTHS_NM[b_idx]
        ax.set_title(f"~{int(actual_wl)} nm (band {b_idx})")
        ax.axis("off")
    fig.suptitle(f"{stem} — real HSI band slices", fontsize=14)
    fig.tight_layout()
    montage_path = out_dir / f"{stem}_montage.png"
    fig.savefig(str(montage_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[extract] saved {montage_path}")

    return {
        "stem": stem,
        "cube_shape": tuple(cube.shape),
        "cube_min": float(cube.min()),
        "cube_max": float(cube.max()),
        "cube_mean": float(cube.mean()),
        "npy_path": str(npy_path),
        "png_paths": [str(p) for p in saved_pngs],
        "montage_path": str(montage_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", default="Train_1000",
                   help="File stem, e.g. Train_1000 or Test_42. Default: Train_1000.")
    p.add_argument("--data-root",
                   default="/mnt/c/Users/jaswa/ollama/Smart_Spectra/data/raw/Agro-HSR/Train_Spec",
                   help="Directory containing <stem>.mat files. "
                        "Default: WSL path to the project's Agro-HSR Train_Spec/. "
                        "On native Windows, override with the corresponding "
                        "C:\\... path.")
    p.add_argument("--out-root", default=None,
                   help="Output directory. Default: ./hsi_samples next to this script.")
    p.add_argument("--stretch", default="per_band_max",
                   choices=["percentile", "full", "per_band_max"],
                   help="Contrast stretch for band PNGs. "
                        "'per_band_max' (default) shows structure clearly; "
                        "'percentile' is robust but can wash out dark cubes; "
                        "'full' preserves absolute scale.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_root = Path(args.out_root) if args.out_root else here / "hsi_samples"
    data_root = Path(args.data_root)

    summary = extract_one(args.stem, data_root, out_root, stretch_mode=args.stretch)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()