"""
Build a manifest of every Vaishnavi/Puvi Lakshmi 2023 .bil file with
its 3-class label and wavelengths, and persist per-cube mean spectra
binned to Agro-HSR's 31-band layout (400-1000 nm @ 20 nm spacing).

Manifest schema (JSON):
{
  "wavelengths_binned_nm": [400, 420, 440, ..., 980, 1000],   # 31 bands
  "wavelength_bin_edges_nm": [...],                          # used for binning
  "cubes": [
    {
      "bil_path": "...path/to/file.bil",
      "pesticide_class": "fresh" | "fungicide" | "insecticide",
      "source_label_raw": "Apple_samples/Monostar/Fresh",  # for traceability
      "shape": [lines, samples, 300],
      "mean_spectrum_31band": [0.064, ...],  # binned reflectance, clipped to [0, 1]
      "mean_spectrum_15visible": [...],       # 400-680 nm subset
      "max_reflectance": 1.0,                # post-clip
      "valid_pixel_fraction": 0.42,          # fraction of pixels in [0.001, 0.99]
    },
    ...
  ]
}

The "valid_pixel_fraction" is the share of pixels with reflectance in a sane
range — used downstream to weight samples (apples that are mostly background
shouldn't dominate training).

This step is intentionally conservative: it loads the cube, computes the
mean spectrum, applies the reflectance scale factor, clips to [0, 1], bins
to 31 bands, and frees memory. We do NOT yet run Model 1.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Reuse the ENVI reader from inspect_vaishnavi.py
sys.path.insert(0, str(Path(__file__).parent))
from inspect_vaishnavi import read_envi_bil  # noqa: E402


# ---------------------------------------------------------------------------
# 3-class label mapping
# ---------------------------------------------------------------------------
# Folder structure (verified):
#   Apple_samples/Monostar/{Fresh,High,Low}        → fresh / insecticide-H / insecticide-L
#   Apple_samples/Nativo/{Fresh,High,Low}          → fresh / fungicide-H / fungicide-L
#   Fungicide_Apple/DIB_Apple                      → fresh
#   Fungicide_Apple/DIB_AppleNativohigh            → fungicide-H
#   Fungicide_Apple/DIB_AppleNativolow             → fungicide-L
#   Pesticide_Apple/DIB_Applemono                  → fresh
#   Pesticide_Apple/DIB_Applemonohigh              → insecticide-H
#   Pesticide_Apple/DIB_Applemonolow               → insecticide-L
#
# Unknown folders (Apple_samples/Monostar/Unknown/{fresh,pesticide_unknownconc})
# are skipped because the pesticide concentration is unlabelled.

# Direct folder-name → class mapping (most reliable: no parent-name dependency)
_FRESH_FOLDERS = {"Fresh", "DIB_Apple", "DIB_Applemono"}
_FUNGICIDE_FOLDERS = {"Nativohigh", "Nativolow", "DIB_AppleNativohigh", "DIB_AppleNativolow"}
_INSECTICIDE_FOLDERS = {"monohigh", "monolow", "DIB_Applemonohigh", "DIB_Applemonolow", "High", "Low"}
# Note: in Apple_samples/Monostar/{High,Low}, the parent "Monostar" disambiguates
# insecticide vs fungicide. Apple_samples/Nativo/{High,Low} → fungicide.
# We use the parent-folder disambiguation only for the Apple_samples subset.


def label_from_path(bil_path: Path) -> tuple[str, str] | None:
    """Return (pesticide_class_3, source_label_raw) for a .bil path, or None if
    the cube's label is unknown (we skip those)."""
    parts = bil_path.parts
    leaf_dir = parts[-2]  # immediate parent of the .bil file

    if leaf_dir in _FRESH_FOLDERS:
        return "fresh", "/".join(parts[-3:])

    if leaf_dir in _FUNGICIDE_FOLDERS:
        return "fungicide", "/".join(parts[-3:])

    if leaf_dir in _INSECTICIDE_FOLDERS:
        # Apple_samples/Monostar/{High,Low} → insecticide
        # Pesticide_Apple/DIB_Apple{mono,monohigh,monolow}
        if any(p in parts for p in ("Monostar", "Pesticide_Apple")):
            return "insecticide", "/".join(parts[-3:])
        if any(p in parts for p in ("Nativo", "Fungicide_Apple")):
            return "fungicide", "/".join(parts[-3:])

    return None  # Unknown folder — skip


# ---------------------------------------------------------------------------
# Bin 300 → 31 bands (Agro-HSR layout: 400-1000 nm @ 20 nm spacing)
# ---------------------------------------------------------------------------
# Constants live in scripts/agro_hsr_bands.py — re-exported here for
# backward compatibility with older code that does
#   `from build_vaishnavi_manifest import AGRO_BANDS, ...`.
from agro_hsr_bands import AGRO_BANDS, AGRO_BAND_EDGES, VISIBLE_BAND_IDX  # noqa: F401, E402


def bin_to_agro(spectrum_300: np.ndarray, wavelengths_300: np.ndarray) -> np.ndarray:
    """Bin a 300-band spectrum (1D, reflectance) to 31 bands using the Agro-HSR
    20-nm grid via trapezoidal integration over each target band edge window.

    Out-of-range wavelengths (<390 or >1010) are dropped.
    """
    out = np.zeros(31, dtype=np.float32)
    for i in range(31):
        lo = AGRO_BAND_EDGES[i]
        hi = AGRO_BAND_EDGES[i + 1]
        # Indices where the source wavelength falls within [lo, hi]
        mask = (wavelengths_300 >= lo) & (wavelengths_300 <= hi)
        if not mask.any():
            # Fallback: nearest band (rare edge case at <400 or >1000)
            nearest = int(np.argmin(np.abs(wavelengths_300 - (lo + hi) / 2)))
            out[i] = float(spectrum_300[nearest])
            continue
        out[i] = float(spectrum_300[mask].mean())
    return out


# ---------------------------------------------------------------------------
# Apple-pixel mask via NIR background heuristic
# ---------------------------------------------------------------------------
def apple_pixel_mask(cube: np.ndarray) -> np.ndarray:
    """Rough apple/non-apple mask. Apples have moderate NIR reflectance
    (~0.3-0.8); backgrounds (dark lab table or pure white reference) tend to
    be either very dark or saturated. Return a boolean mask of likely-apple
    pixels."""
    # Use the mean of the upper-third of bands as the NIR proxy (rough)
    nir_start = int(cube.shape[2] * 2 / 3)
    nir = cube[:, :, nir_start:].astype(np.float32).mean(axis=2)
    # Apple NIR is moderate; very dark or very bright pixels are background
    return (nir > 0.05) & (nir < 0.95)


# ---------------------------------------------------------------------------
# Per-cube processing
# ---------------------------------------------------------------------------
@dataclass
class CubeEntry:
    bil_path: str
    pesticide_class: str
    source_label_raw: str
    shape: list[int]
    mean_spectrum_31band: list[float]
    mean_spectrum_15visible: list[float]
    max_reflectance: float
    valid_pixel_fraction: float


def process_cube(bil_path: Path) -> CubeEntry | None:
    label = label_from_path(bil_path)
    if label is None:
        return None
    cls, src_label = label

    cube, hdr = read_envi_bil(bil_path)
    wavelengths = hdr["wavelengths"]
    scale = float(hdr.get("reflectance scale factor", 1.0))

    # Apply reflectance scale, clip to [0, 1], keep float32
    refl = cube.astype(np.float32) / scale
    refl = np.clip(refl, 0.0, 1.0)

    # Mask apple pixels
    mask = apple_pixel_mask(refl)
    valid_frac = float(mask.mean())
    if not mask.any():
        return None

    # Mean spectrum across apple pixels only
    mean_spec = refl[mask].mean(axis=0)  # (300,)

    # Bin to 31 bands
    spec_31 = bin_to_agro(mean_spec, wavelengths)
    spec_15 = spec_31[VISIBLE_BAND_IDX]

    return CubeEntry(
        bil_path=str(bil_path),
        pesticide_class=cls,
        source_label_raw=src_label,
        shape=list(cube.shape),
        mean_spectrum_31band=spec_31.tolist(),
        mean_spectrum_15visible=spec_15.tolist(),
        max_reflectance=float(refl[mask].max()),
        valid_pixel_fraction=valid_frac,
    )


def main() -> int:
    root = Path(
        "/mnt/c/Users/jaswa/ollama/Smart_spectra/data/pesticides/vaishnavi_2023"
    )
    bils = sorted(root.rglob("*.bil"))
    print(f"Found {len(bils)} .bil files")

    entries: list[CubeEntry] = []
    skipped = 0
    for i, p in enumerate(bils):
        e = process_cube(p)
        if e is None:
            skipped += 1
            print(f"  SKIP: {p.relative_to(root)}")
            continue
        entries.append(e)
        if (i + 1) % 5 == 0:
            print(f"  processed {i+1}/{len(bils)}")

    # Manifest
    manifest = {
        "wavelengths_binned_nm": AGRO_BANDS.tolist(),
        "wavelength_bin_edges_nm": AGRO_BAND_EDGES.tolist(),
        "visible_band_idx": VISIBLE_BAND_IDX,
        "class_to_idx": {"fresh": 0, "fungicide": 1, "insecticide": 2},
        "idx_to_class": ["fresh", "fungicide", "insecticide"],
        "n_cubes_kept": len(entries),
        "n_cubes_skipped": skipped,
        "cubes": [asdict(e) for e in entries],
    }
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {out}")
    print(f"Kept {len(entries)} cubes, skipped {skipped}")

    # Class breakdown
    from collections import Counter
    counts = Counter(e.pesticide_class for e in entries)
    print(f"Class counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())