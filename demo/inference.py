"""
Model 1 inference wrapper for the Streamlit demo.

Loads a fine-tuned MST++ checkpoint and reconstructs a 31-band
hyperspectral cube (400-1000 nm) from a single RGB fruit photo.

Decoupled from `scripts/infer_mst.py` so the per-image min-max
preprocessing (the bug fix) lives entirely in this module. The
earlier /255-normalized path produced near-zero output
(mean ~0.0000, range +/-0.004) — DO NOT reintroduce it.

What this module provides:
    * DEFAULT_CHECKPOINT  — path to net_39epoch.pth (overridable).
    * load_model(ckpt, device)        — constructs MST_Plus_Plus and
      loads state_dict. Tries strict=True first; falls back to
      strict=False only on failure with a loud warning.
    * preprocess_image(image_path)    — per-image min-max
      normalization matching hsi_dataset.py (NOT /255).
    * run_model1(image_path, model, device)
      — runs the forward pass and returns the cube as (H, W, 31).

CLI smoke-test:
    python inference.py --rgb <photo.jpg> --ckpt <net_39epoch.pth>
                        --out /tmp/cube.npy
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
try:
    import shap
except ImportError:
    shap = None
from scipy.signal import savgol_filter

class SpectralMLP(nn.Module):
    def __init__(self, input_dim=312):
        super(SpectralMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------
# Path setup — let `from MST_Plus_Plus import MST_Plus_Plus` resolve.
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ARCH_DIR = (
    PROJECT_DIR
    / "code"
    / "MST-plus-plus"
    / "train_code"
    / "architecture"
)

def _disable_cuda_on_tensor() -> None:
    def _safe_cuda(self, *args, **kwargs):
        return self
    torch.Tensor.cuda = _safe_cuda

_disable_cuda_on_tensor()

if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

from MST_Plus_Plus import MST_Plus_Plus

# ---------------------------------------------------------------------
# Default checkpoint resolver.
# ---------------------------------------------------------------------
CHECKPOINTS_ROOT = (
    PROJECT_DIR / "models" / "checkpoints"
)

PINNED_DEFAULT_CKPT = (
    CHECKPOINTS_ROOT / "run_trained" / "net_best.pth"
)

def _latest_net_best(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    for pattern in ("net_best.pth", "net_*epoch.pth"):
        candidates = sorted(
            (p for p in root.rglob(pattern)
             if not p.parts[len(root.parts):]
                  or "run_dummy" not in Path(*p.parts[len(root.parts):]).parts),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None

def _resolve_default_checkpoint() -> Path:
    env = os.environ.get("SMARTSPECTRA_CKPT", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"SMARTSPECTRA_CKPT={env!r} does not point to an existing file.")

    if PINNED_DEFAULT_CKPT.is_file():
        return PINNED_DEFAULT_CKPT

    found = _latest_net_best(CHECKPOINTS_ROOT)
    if found is not None:
        return found
    raise FileNotFoundError("No usable checkpoint found.")

def default_checkpoint_path() -> Path:
    return _resolve_default_checkpoint()

class _LazyCheckpointPath:
    def __str__(self) -> str:
        return str(_resolve_default_checkpoint())
    def __fspath__(self) -> str:
        return str(self)
    def __repr__(self) -> str:
        return repr(str(self))
    def __path__(self) -> Path:
        return _resolve_default_checkpoint()

DEFAULT_CHECKPOINT = _LazyCheckpointPath()

def _load_state_dict_flexible(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint type {type(obj).__name__} at {path}")

    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    for key in ("state_dict", "model", "model_state_dict", "net"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
    if tensor_keys:
        return {k: obj[k] for k in tensor_keys}

    raise ValueError(f"Could not find a state_dict in checkpoint at {path}.")

def load_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = MST_Plus_Plus()

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    ckpt_p = Path(checkpoint_path)

    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}.")

    sd = _load_state_dict_flexible(ckpt_p)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}.")
    return preprocess_image_from_array(bgr, is_bgr=True)

def preprocess_image_from_array(img_arr: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    if img_arr.shape[0] != 256 or img_arr.shape[1] != 256:
        img_arr = cv2.resize(img_arr, (256, 256), interpolation=cv2.INTER_AREA)
    arr = img_arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    if is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    rgb = np.transpose(arr, [2, 0, 1])
    rgb = np.expand_dims(rgb, axis=0)
    return rgb.astype(np.float32)

def run_model1(
    image_input: str | Path | np.ndarray,
    model: nn.Module,
    device: torch.device | str | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    if isinstance(image_input, (str, Path)):
        arr = preprocess_image(image_input)
    else:
        arr = preprocess_image_from_array(image_input, is_bgr=False)

    x = torch.from_numpy(arr).to(device)

    with torch.no_grad():
        y = model(x)

    cube = y.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return cube

def enhance_for_display(
    cube: np.ndarray,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
) -> np.ndarray:
    lo = np.percentile(cube, percentile_lo)
    hi = np.percentile(cube, percentile_hi)
    stretched = (cube - lo) / (hi - lo + 1e-8)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)

def get_regional_mean(cube: np.ndarray, mask: np.ndarray, region_size: float = 0.2) -> np.ndarray:
    if not mask.any():
        return np.zeros(cube.shape[2], dtype=np.float32)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rh = rmax - rmin
    cw = cmax - cmin
    ry_s = rmin + int(rh * (1 - region_size) / 2)
    ry_e = rmax - int(rh * (1 - region_size) / 2)
    cx_s = cmin + int(cw * (1 - region_size) / 2)
    cx_e = cmax - int(cw * (1 - region_size) / 2)

    region_cube = cube[ry_s:ry_e, cx_s:cx_e, :]
    region_mask = mask[ry_s:ry_e, cx_s:cx_e]

    if region_mask.any():
        return region_cube[region_mask].mean(axis=0).astype(np.float32)
    else:
        return region_cube.mean(axis=(0, 1)).astype(np.float32)

def get_fresh_apple_baseline() -> np.ndarray:
    wavelengths = np.linspace(400, 1000, 31)
    baseline = 0.1 + 0.4 * (1 - np.exp(-(wavelengths - 400) / 200))
    dip = 0.05 * np.exp(-((wavelengths - 970)**2) / (2 * 20**2))
    baseline -= dip
    return baseline.astype(np.float32)

def compute_spectral_ratios(mean_spec: np.ndarray) -> np.ndarray:
    spec = mean_spec + 1e-8
    ratios = [
        spec[30] / spec[0],
        spec[15] / spec[0],
        spec[30] / spec[15],
        (spec[30] - spec[15]) / (spec[30] + spec[15]),
        spec[20] / spec[10],
        spec[25] / spec[5],
        spec[30] / spec[20],
        spec[28] / spec[22],
        spec[30] / spec[26],
        (spec[30] - spec[28]) / (spec[30] + spec[28]),
        spec[24] / spec[18],
        spec[10] / spec[5],
    ]
    return np.array(ratios, dtype=np.float32)

def segment_produce(
    rgb_u8: np.ndarray,
    min_saturation: float = 0.15,
    min_value: float = 0.15,
    max_value: float = 0.95,
) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    V = np.max(rgb, axis=-1)
    Min = np.min(rgb, axis=-1)
    S = (V - Min) / (V + 1e-6)
    return (S > min_saturation) & (V > min_value) & (V < max_value)

def segment_fruit(
    rgb_u8: np.ndarray,
    *,
    red_margin_g: float = 0.06,
    red_margin_b: float = 0.10,
    min_red: float = 0.30,
    min_saturation: float = 0.25,
    min_value: float = 0.25,
) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    V = np.max(rgb, axis=-1)
    Min = np.min(rgb, axis=-1)
    S = (V - Min) / (V + 1e-6)
    is_red = (R > G + red_margin_g) & (R > B + red_margin_b) & (R > min_red)
    is_sat = S > min_saturation
    is_bright = V > min_value
    return is_red & is_sat & is_bright

def cube_fruit_stats(cube: np.ndarray, mask: np.ndarray) -> dict:
    if mask.shape[:2] != cube.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match cube spatial shape {cube.shape[:2]}")
    if not mask.any():
        return {
            "n_pixels": 0,
            "mean": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "band_means": np.zeros(cube.shape[2], dtype=np.float32),
            "per_band_max": np.zeros(cube.shape[2], dtype=np.float32),
            "per_band_std": np.zeros(cube.shape[2], dtype=np.float32),
            "empty": True,
        }
    px = cube[mask]
    return {
        "n_pixels": int(mask.sum()),
        "mean": float(px.mean()),
        "max": float(px.max()),
        "std": float(px.std()),
        "band_means": px.mean(axis=0).astype(np.float32),
        "per_band_max": px.max(axis=0).astype(np.float32),
        "per_band_std": px.std(axis=0).astype(np.float32),
        "empty": False,
    }

def snv(spectrum: np.ndarray) -> np.ndarray:
    std = np.std(spectrum)
    if std == 0: return spectrum
    return (spectrum - np.mean(spectrum)) / std

def apply_sg_derivative(spectrum: np.ndarray, window=5, order=2) -> np.ndarray:
    """Savitzky-Golay 1st derivative.
    Uses np.diff to ensure output length is len(spectrum) - 1, matching training data.
    """
    return np.diff(savgol_filter(spectrum, window, order))

def load_model2_hybrid() -> tuple[any, any]:
    ckpt_path = PROJECT_DIR / "models" / "checkpoints" / "model2_hybrid.pkl"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model 2 Hybrid checkpoint not found at {ckpt_path}")
    return joblib.load(str(ckpt_path))

def load_model2_ultra() -> tuple[any, any]:
    ckpt_path = PROJECT_DIR / "models" / "checkpoints" / "model2_ultra.pkl"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model 2 Ultra checkpoint not found at {ckpt_path}")
    return joblib.load(str(ckpt_path))

def predict_pesticide(cube: np.ndarray, model_checkpoint: dict, mask: np.ndarray | None = None) -> tuple[str, float]:
    model = model_checkpoint.get("model", model_checkpoint.get("model_s1"))
    scaler = model_checkpoint["scaler"]

    if mask is not None and mask.any():
        px = cube[mask]
        mean_spec = px.mean(axis=0)
        std_spec = px.std(axis=0)
    else:
        mean_spec = cube.mean(axis=(0, 1))
        std_spec = cube.std(axis=(0, 1))

    snv_mean = snv(mean_spec)
    sg_mean = apply_sg_derivative(snv_mean)

    features = np.concatenate([mean_spec, snv_mean, sg_mean, std_spec])
    features = features.reshape(1, -1)

    features_scaled = scaler.transform(features)
    pred_class = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]

    label_map = {0: "Fresh", 1: "Fungicide", 2: "Insecticide"}
    return label_map[pred_class], float(probs[pred_class])

def predict_pesticide_hybrid(cube: np.ndarray, rgb_u8: np.ndarray, model_checkpoint: dict, mask: np.ndarray | None = None) -> tuple[dict, str, float, np.ndarray]:
    model = model_checkpoint.get("model", model_checkpoint.get("model_s1"))
    scaler = model_checkpoint["scaler"]

    img_f = rgb_u8.astype(np.float32) / 255.0
    if mask is not None and mask.any():
        rgb_pixels = img_f[mask]
        rgb_mean = np.mean(rgb_pixels, axis=0)
        rgb_std = np.std(rgb_pixels, axis=0)
    else:
        rgb_mean = np.mean(img_f, axis=(0, 1))
        rgb_std = np.std(img_f, axis=(0, 1))
    rgb_features = np.concatenate([rgb_mean, rgb_std])

    if mask is not None and mask.any():
        px = cube[mask]
        mean_spec = px.mean(axis=0)
        std_spec = px.std(axis=0)
    else:
        mean_spec = cube.mean(axis=(0, 1))
        std_spec = cube.std(axis=(0, 1))

    snv_mean = snv(mean_spec)
    sg_mean = apply_sg_derivative(snv_mean)
    ratio_features = compute_spectral_ratios(mean_spec)

    features = np.concatenate([snv_mean, sg_mean, std_spec, ratio_features]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    pred_class = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]

    if len(probs) == 2:
        label_map = {0: "Fresh", 1: "Pesticide"}
    else:
        label_map = {0: "Fresh", 1: "Fungicide", 2: "Insecticide"}

    prob_map = {label_map[i]: float(probs[i]) for i in range(len(probs))}

    prob_map = {label_map[i]: float(probs[i]) for i in range(len(probs))}

    return prob_map, label_map[pred_class], float(probs[pred_class]), features_scaled

def extract_zonal_pure_science_features(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Divides the apple into 3 concentric zones (Center, Mid, Edge) and extracts
    the Pure Science feature vector for each. Total features: 3 * 104 = 312.
    """
    def get_features(c, m):
        if not m.any(): return np.zeros(104, dtype=np.float32)
        px = c[m]
        mean_spec, std_spec = px.mean(axis=0), px.std(axis=0)
        snv_mean = snv(mean_spec)
        sg_mean = apply_sg_derivative(snv_mean)
        ratio_features = compute_spectral_ratios(mean_spec)
        return np.concatenate([np.concatenate([snv_mean, sg_mean, std_spec]), ratio_features])

    # 1. Center Zone (Central 30%)
    center_mask = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.15), int(w * 0.15)
    center_mask[cy-dy:cy+dy, cx-dx:cx+dx] = True
    center_mask &= mask
    feat_center = get_features(cube, center_mask)

    # 2. Mid Zone (Between 30% and 60%)
    mid_mask = np.zeros_like(mask, dtype=bool)
    dy_m, dx_m = int(h * 0.3), int(w * 0.3)
    mid_mask[cy-dy_m:cy+dy_m, cx-dx_m:cx+dx_m] = True
    mid_mask &= mask
    mid_mask &= ~center_mask
    feat_mid = get_features(cube, mid_mask)

    # 3. Edge Zone (Everything else)
    edge_mask = mask & ~mid_mask & ~center_mask
    feat_edge = get_features(cube, edge_mask)

    return np.concatenate([feat_center, feat_mid, feat_edge])

def get_spectral_fingerprint(feat_scaled, model_data):

    """
    Computes SHAP values to identify which spectral bands drove the prediction.
    """
    model = model_data["model_s1"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feat_scaled)

    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]

    feature_names = []
    for i in range(31): feature_names.append(f"SNV_Band_{i+1}")
    for i in range(30): feature_names.append(f"SG_Deriv_{i+1}")
    for i in range(31): feature_names.append(f"STD_Band_{i+1}")
    for i in range(12): feature_names.append(f"Ratio_{i+1}")

    fingerprint = []
    for name, val in zip(feature_names, vals):
        fingerprint.append((name, float(val)))

    return fingerprint

def resolve_checkpoint_for_pipeline(pipeline: str) -> Path:
    PIPELINE_TO_CHECKPOINT = {
        "A":         PROJECT_DIR / "models/checkpoints/run_trained/net_best.pth",
        "B":         PROJECT_DIR / "models/checkpoints/run_trained/net_best.pth",
        "C":         PROJECT_DIR / "models/checkpoints/run_apple_spectrofood_finetuned/net_best.pth",
        "fallback":  PROJECT_DIR / "models/checkpoints/run_trained/net_best.pth",
    }
    if pipeline not in PIPELINE_TO_CHECKPOINT:
        raise KeyError(f"Unknown pipeline {pipeline!r}")
    p = PIPELINE_TO_CHECKPOINT[pipeline]
    if not p.is_file():
        raise FileNotFoundError(f"Pipeline {pipeline!r} -> checkpoint {p} does not exist.")
    return p

def route_and_infer(
    image_path: str | Path,
    *,
    model: nn.Module | None = None,
    device: torch.device | str | None = None,
    skip_inference: bool = False,
) -> tuple[any, np.ndarray | None]:
    from routing import route_image, RoutingDecision
    decision = route_image(image_path)

    if decision.tier == 3 or decision.pipeline_to_run is None or skip_inference:
        return decision, None

    ckpt = resolve_checkpoint_for_pipeline(decision.pipeline_to_run)

    if model is None or device is None:
        loaded_model, loaded_device = load_model(ckpt, device)
        model = model or loaded_model
        device = device or loaded_device

    cube = run_model1(image_path, model, device)
    return decision, cube

def list_test_pairs() -> list[str]:
    _TEST_LAYOUTS = (
        (PROJECT_DIR / "data" / "raw" / "Test_RGB",
         PROJECT_DIR / "data" / "raw" / "Test_Spec"),
        (PROJECT_DIR / "data" / "raw" / "Agro-HSR" / "Test_RGB",
         PROJECT_DIR / "data" / "raw" / "Agro-HSR" / "Test_Spec"),
    )
    _resolved = None
    for rgb, spec in _TEST_LAYOUTS:
        if rgb.is_dir() and spec.is_dir():
            _resolved = (rgb, spec)
            break
    if not _resolved: return []
    rgb_dir, spec_dir = _resolved
    stems = []
    for jpg in sorted(glob.glob(str(rgb_dir / "*.jpg"))):
        stem = Path(jpg).stem
        if (spec_dir / f"{stem}.mat").is_file():
            stems.append(stem)
    return stems

def load_ground_truth_hsi(stem: str) -> np.ndarray:
    import hdf5storage
    TEST_SPEC_DIR = (PROJECT_DIR / "data" / "raw" / "Test_RGB") # simplified for logic
    mat_path = TEST_SPEC_DIR / f"{stem}.mat"
    mat = hdf5storage.loadmat(str(mat_path))
    return mat["cube"].astype(np.float32)

def crop_to_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h_g, w_g = gt.shape[:2]
    h_p, w_p = pred.shape[:2]
    if (h_g, w_g) == (h_p, w_p): return gt
    top = max(0, (h_g - h_p) // 2)
    left = max(0, (w_g - w_p) // 2)
    return gt[top:top + h_p, left:left + w_p, :]

def compute_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-3) -> dict:
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)
    diff = p - g
    abs_diff = np.abs(diff)
    mrae = float(np.mean(abs_diff / (np.abs(g) + eps)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    p_lo, p_hi = float(p.min()), float(p.max())
    if p_hi - p_lo < 1e-8: psnr = 0.0
    else:
        p_rescaled = (p - p_lo) / (p_hi - p_lo)
        g_rescaled = np.clip(g, 0.0, 1.0)
        mse = float(np.mean((p_rescaled - g_rescaled) ** 2))
        psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else 100.0
    return {"mrae": mrae, "rmse": rmse, "psnr": psnr}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", required=True)
    p.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--out", default="output_cube.npy")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_model(args.ckpt, device)
    cube = run_model1(args.rgb, model, device)
    np.save(args.out, cube)
