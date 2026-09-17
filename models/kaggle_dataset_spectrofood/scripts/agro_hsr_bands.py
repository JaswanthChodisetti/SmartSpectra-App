"""
Agro-HSR band layout — the 31-band spectral convention used across
this project.

The target output for MST++ fine-tuning and inference is the 31-band
Agro-HSR layout (400-1000 nm @ 20 nm spacing). All data preparation
scripts (Vaishnavi binning, SpectroFood binning, CIE 1931 synthesis)
should use these constants to ensure cubes line up at training time.

Layout:
    AGRO_BANDS      — 31 band centers in nm
    AGRO_BAND_EDGES — 32 bin edges (390 to 1010 nm)
    VISIBLE_BAND_IDX — indices of the visible bands (0-14, 400-680 nm)

Use:
    from agro_hsr_bands import AGRO_BANDS, AGRO_BAND_EDGES, VISIBLE_BAND_IDX

This file was split out of build_vaishnavi_manifest.py — that module
still re-exports the constants for backward compatibility but new code
should import from here.
"""
from __future__ import annotations

import numpy as np


AGRO_BANDS = np.linspace(400, 1000, 31)  # 31 target centers
AGRO_BAND_EDGES = np.concatenate([
    [AGRO_BANDS[0] - 10],                          # 390
    (AGRO_BANDS[:-1] + AGRO_BANDS[1:]) / 2,        # midpoints
    [AGRO_BANDS[-1] + 10],                         # 1010
])
# 32 edges from 390 to 1010 in 20-nm steps.
VISIBLE_BAND_IDX = list(range(0, 15))  # 400-680 nm = first 15 of 31 bands