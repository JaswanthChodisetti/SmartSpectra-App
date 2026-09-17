"""
Shared CIE 1931 -> sRGB synthesis module.

Converts per-pixel spectra (reflectance, arbitrary wavelength grid) into
RGB images using the CIE 1931 2-degree color matching functions (CMFs)
and the sRGB IEC 61966-2-1 encoding.

Used by:
  - scripts/render_fake_rgb.py (Vaishnavi 300-band cubes -> RGB)
  - scripts/train_model2_deployment.py (after refactor)
  - scripts/synthesize_apple_adaptation_pairs.py (Puvi Lakshmi -> RGB
    for Model 1 apple-domain fine-tuning; Option B in the report).

Inputs:
  spectra: (H, W, N) numpy array, reflectance in [0, 1].
  wavelengths: (N,) numpy array of wavelength centers in nm, monotonically
    increasing. May have arbitrary spacing (linear or not).

Outputs:
  rgb_uint8: (H, W, 3) uint8 array. sRGB-encoded, range [0, 255].

Method (5-nm CIE 1931 2-deg standard tables):
  1. Resample spectra to the 5-nm CMF grid (390-810 nm from the user CSV).
  2. Trapezoidal-integrate against x_bar, y_bar, z_bar -> XYZ per pixel.
  3. White-point normalize so equal-energy illuminant maps to neutral.
  4. Linear sRGB matrix (IEC 61966-2-1).
  5. sRGB gamma 2.4 piecewise, scale to uint8.

Reference: http://cvrl.ioo.ucl.ac.uk/cmfs.htm (CIE 1931 2-deg, 5 nm).
Data: loaded at import time from data/cie_xyz_5nm.csv (user-supplied).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from load_cie_xyz import load_cie_xyz


# ---------------------------------------------------------------------------
# CIE 1931 2-deg CMFs at 5 nm spacing, loaded from data/cie_xyz_5nm.csv.
# The CSV covers 390-810 nm at 5 nm (85 entries); the standard table goes
# to 830 but the missing 815-830 rows are below the visible range and have
# y_bar values very close to zero, so the visible-range synthesis is unaffected.
# ---------------------------------------------------------------------------
_CMFS_NM, _X_BAR, _Y_BAR, _Z_BAR = load_cie_xyz()


# ---------------------------------------------------------------------------
# D65 white-point XYZ and XYZ->linear sRGB matrix (IEC 61966-2-1).
# ---------------------------------------------------------------------------
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883])
_XYZ_TO_LINEAR_SRGB = np.array([
    [ 3.2406255, -1.5372080, -0.4986286],
    [-0.9689307,  1.8757561,  0.0415175],
    [ 0.0557101, -0.2040211,  1.0569959],
])


def _srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB transfer function. Linear [0, 1] -> nonlinear."""
    threshold = 0.0031308
    a = 0.055
    low = 12.92 * linear
    high = (1 + a) * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - a
    return np.where(linear <= threshold, low, high)


def spectra_to_rgb(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Convert (H, W, N) reflectance spectra to (H, W, 3) uint8 RGB (sRGB).

    Args:
        spectra: (H, W, N), reflectance in [0, 1].
        wavelengths: (N,), wavelength centers in nm, monotonically
            increasing. Arbitrary spacing.

    Returns:
        rgb_uint8: (H, W, 3) uint8, sRGB-encoded, range [0, 255].

    White-point normalization: a perfect reflector (reflectance 1.0 at all
    wavelengths under an equal-energy illuminant) maps to neutral RGB (1, 1, 1)
    after the sRGB transfer. This is the standard relative-XYZ formulation
    used in colour-science libraries.
    """
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be (H, W, N), got {spectra.shape}")
    if spectra.shape[2] != wavelengths.shape[0]:
        raise ValueError(
            f"spectra has {spectra.shape[2]} bands but wavelengths has "
            f"{wavelengths.shape[0]}"
        )
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be monotonically increasing")

    # Interpolate onto the 5 nm CMF grid (linear, fill_value=0 outside).
    interp = interp1d(
        wavelengths, spectra.astype(np.float32),
        axis=2, kind="linear", bounds_error=False, fill_value=0.0,
    )
    spec_on_grid = interp(_CMFS_NM)  # (H, W, 95)

    # Trapezoidal integration against each CMF -> XYZ. dx is the spacing
    # between adjacent CMF rows (= 5 nm per the loaded table).
    dx = float(_CMFS_NM[1] - _CMFS_NM[0])
    X = np.trapezoid(spec_on_grid * _X_BAR, dx=dx, axis=2)
    Y = np.trapezoid(spec_on_grid * _Y_BAR, dx=dx, axis=2)
    Z = np.trapezoid(spec_on_grid * _Z_BAR, dx=dx, axis=2)

    # Equal-energy illuminant (white for this normalization scheme).
    # White point under EE = (sum(X_BAR), sum(Y_BAR), sum(Z_BAR)) * dx
    EE_X = float(np.trapezoid(_X_BAR, dx=dx))
    EE_Y = float(np.trapezoid(_Y_BAR, dx=dx))
    EE_Z = float(np.trapezoid(_Z_BAR, dx=dx))
    # Scale so Y_white = 1 (relative XYZ)
    X = X / EE_Y
    Y = Y / EE_Y
    Z = Z / EE_Y

    # Convert to linear sRGB via matrix, normalize so EE white -> (1,1,1).
    XYZ_stack = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    linear_srgb = XYZ_stack @ _XYZ_TO_LINEAR_SRGB.T  # (H, W, 3)
    # EE white's linear sRGB value
    ee_xyz = np.array([EE_X, EE_Y, EE_Z]) / EE_Y
    ee_linear_srgb = _XYZ_TO_LINEAR_SRGB @ ee_xyz  # shape (3,)
    # Normalize per-channel so EE white becomes (1, 1, 1)
    linear_srgb = linear_srgb / ee_linear_srgb

    # Clip negatives (out-of-gamut), apply gamma, scale to uint8
    linear_srgb = np.clip(linear_srgb, 0.0, 1.0)
    nonlinear = _srgb_gamma(linear_srgb)
    return (np.clip(nonlinear, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Smoke test: a flat 1.0 reflectance spectrum should produce (255, 255, 255)
# after white-point normalization.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test = np.ones((1, 1, len(_CMFS_NM)), dtype=np.float32)
    rgb = spectra_to_rgb(test, _CMFS_NM)
    print(f"Flat 1.0 reflectance -> RGB: {rgb.flatten()} (expected near 255,255,255)")
    assert np.all(rgb >= 250), f"flat-1.0 should be ~white, got {rgb}"

    # Pure red (660 nm narrow peak)
    red_test = np.zeros_like(test)
    # 5 nm peak at 660 nm -> index = (660-360)/5 = 60
    red_test[0, 0, 60] = 1.0
    rgb_red = spectra_to_rgb(red_test, _CMFS_NM)
    print(f"Narrow peak at 660 nm -> RGB: {rgb_red.flatten()} "
          f"(R should dominate)")
    assert rgb_red[0, 0, 0] > rgb_red[0, 0, 1], "R should dominate at 660 nm"

    # Pure green (555 nm, the y_bar peak)
    grn_test = np.zeros_like(test)
    # Find the index closest to 555 nm
    idx_555 = int(np.argmin(np.abs(_CMFS_NM - 555)))
    grn_test[0, 0, idx_555] = 1.0
    rgb_grn = spectra_to_rgb(grn_test, _CMFS_NM)
    print(f"Narrow peak at {_CMFS_NM[idx_555]:.0f} nm (y_bar peak) -> RGB: "
          f"{rgb_grn.flatten()}")
    assert rgb_grn[0, 0, 1] > rgb_grn[0, 0, 0], \
        f"G should dominate at {_CMFS_NM[idx_555]:.0f} nm, got {rgb_grn.flatten()}"

    print("\nCIE 1931 -> sRGB smoke test passed.")