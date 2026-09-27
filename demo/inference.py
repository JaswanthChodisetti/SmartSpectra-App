"""
Inference pipeline for the SmartSpectra system.
Handles RGB -> HSI reconstruction using MST++ and extracts
spectral features for pesticide classification.
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

# Setup project paths to ensure architecture imports resolve regardless of OS
# Using Pathlib for cross-platform compatibility (Windows/Linux/Mac)
BASE_DIR = Path(__file__).resolve().parent.parent
ARCH_PATH = BASE_DIR / "code" / "MST-plus-plus" / "train_code" / "architecture"

def _patch_cuda():
    # Prevent crashes on systems without NVIDIA GPUs by overriding .cuda()
    def _safe_cuda(self, *args, **kwargs):
        return self
    torch.Tensor.cuda = _safe_cuda

_patch_cuda()

if str(ARCH_PATH) not in sys.path:
    sys.path.insert(0, str(ARCH_PATH))

from MST_Plus_Plus import MST_Plus_Plus

# Model storage locations
CKPT_DIR = BASE_DIR / "models" / "checkpoints"
DEFAULT_MODEL_PATH = CKPT_DIR / "run_trained" / "net_best.pth"

def _find_best_checkpoint(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    # Look for the best weights or latest epoch
    for pattern in ("net_best.pth", "net_*epoch.pth"):
        matches = sorted(
            (p for p in root.rglob(pattern)
             if "run_dummy" not in str(p)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None

def resolve_ckpt():
    # Check environment variable first, then pinned path, then search
    env_path = os.environ.get("SMARTSPECTRA_CKPT", "").strip()
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    found = _find_best_checkpoint(CKPT_ROOT if 'CKPT_ROOT' in globals() else CKPT_DIR)
    if found:
        return found
    raise FileNotFoundError("Could not locate any valid model weights in checkpoints folder.")

class _LazyPath:
    def __str__(self) -> str: return str(resolve_ckpt())
    def __fspath__(self) -> str: return str(self)
    def __repr__(self) -> str: return repr(str(self))
    def __path__(self) -> Path: return resolve_ckpt()

DEFAULT_CHECKPOINT = _LazyPath()

def _load_weights(path: Path) -> dict:
    # Flexible loading to handle different save formats (dict vs state_dict)
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} is not a dictionary.")

    if all(isinstance(v, torch.Tensor) for v in state.values()):
        return state

    for key in ("state_dict", "model", "model_state_dict", "net"):
        if key in state and isinstance(state[key], dict):
            return state[key]

    t_keys = [k for k, v in state.items() if isinstance(v, torch.Tensor)]
    if t_keys:
        return {k: state[k] for k in t_keys}

    raise ValueError(f"No valid state_dict found in {path}.")

def load_model(checkpoint_path: str | Path | None = None, device: str | torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    net = MST_Plus_Plus()

    target_path = Path(checkpoint_path if checkpoint_path else DEFAULT_CHECKPOINT)
    if not target_path.is_file():
        raise FileNotFoundError(f"Weights not found: {target_path}")

    weights = _load_weights(target_path)
    # Strip 'module.' prefix if saved via DataParallel
    weights = {(k[7:] if k.startswith("module.") else k): v for k, v in weights.items()}

    try:
        net.load_state_dict(weights, strict=True)
    except RuntimeError:
        net.load_state_dict(weights, strict=False)

    net.to(device).eval()
    return net, device

def preprocess_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return preprocess_from_array(img, is_bgr=True)

def preprocess_from_array(img: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    # Ensure 256x256 for MST++
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Use per-image min-max scaling to fix baseline shift
    arr = img.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    if is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # HWC -> CHW and add batch dim
    rgb = np.transpose(arr, [2, 0, 1])
    return np.expand_dims(rgb, axis=0).astype(np.float32)

def run_model1(image_in: str | Path | np.ndarray, model: nn.Module, device=None):
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    arr = preprocess_image(image_in) if isinstance(image_in, (str, Path)) else preprocess_from_array(image_in, is_bgr=False)
    x = torch.from_numpy(arr).to(device)

    with torch.no_grad():
        out = model(x)

    return out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

def enhance_cube(cube: np.ndarray, lo=1.0, hi=99.0):
    # Stretch contrast for better visualization
    v_lo, v_hi = np.percentile(cube, [lo, hi])
    stretched = (cube - v_lo) / (v_hi - v_lo + 1e-8)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)

def get_regional_mean(cube: np.ndarray, mask: np.ndarray, size=0.2):
    if not mask.any():
        return np.zeros(cube.shape[2], dtype=np.float32)

    # Crop to minimal bounding box of the fruit
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Focus on the center region to avoid edge noise
    rh, cw = rmax - rmin, cmax - cmin
    ry_s = rmin + int(rh * (1 - size) / 2)
    ry_e = rmax - int(rh * (1 - size) / 2)
    cx_s = cmin + int(cw * (1 - size) / 2)
    cx_e = cmax - int(cw * (1 - size) / 2)

    region_cube = cube[ry_s:ry_e, cx_s:cx_e, :]
    region_mask = mask[ry_s:ry_e, cx_s:cx_e]

    if region_mask.any():
        return region_cube[region_mask].mean(axis=0).astype(np.float32)
    return region_cube.mean(axis=(0, 1)).astype(np.float32)

def get_baseline_spec() -> np.ndarray:
    # Simplified physical model for a fresh apple baseline
    w = np.linspace(400, 1000, 31)
    base = 0.1 + 0.4 * (1 - np.exp(-(w - 400) / 200))
    # Add the characteristic water dip around 970nm
    base -= 0.05 * np.exp(-((w - 970)**2) / (2 * 20**2))
    return base.astype(np.float32)

def compute_spectral_ratios(spec: np.ndarray) -> np.ndarray:
    s = spec + 1e-8
    # Custom ratios based on chemical absorbance peaks
    return np.array([
        s[30]/s[0], s[15]/s[0], s[30]/s[15], (s[30]-s[15])/(s[30]+s[15]),
        s[20]/s[10], s[25]/s[5], s[30]/s[20], s[28]/s[22],
        s[30]/s[26], (s[30]-s[28])/(s[30]+s[28]), s[24]/s[18], s[10]/s[5]
    ], dtype=np.float32)

def segment_produce(rgb_u8: np.ndarray, sat=0.15, val_min=0.15, val_max=0.95):
    # Basic HSV-like segmentation to isolate fruit from background
    rgb = rgb_u8.astype(np.float32) / 255.0
    v = np.max(rgb, axis=-1)
    min_v = np.min(rgb, axis=-1)
    s = (v - min_v) / (v + 1e-6)
    return (s > sat) & (v > val_min) & (v < val_max)

def snv(spec: np.ndarray) -> np.ndarray:
    # Standard Normal Variate to remove additive/multiplicative scattering
    std = np.std(spec)
    return (spec - np.mean(spec)) / std if std != 0 else spec

def apply_sg_derivative(spec: np.ndarray, win=5, order=2):
    # 1st Derivative to highlight inflection points in the spectrum
    return np.diff(savgol_filter(spec, win, order))

def predict_pesticide(cube: np.ndarray, model_data: dict, mask=None):
    model = model_data.get("model", model_data.get("model_s1"))
    scaler = model_data["scaler"]

    if mask is not None and mask.any():
        px = cube[mask]
        m, s = px.mean(axis=0), px.std(axis=0)
    else:
        m, s = cube.mean(axis=(0, 1)), cube.std(axis=(0, 1))

    # The Pure Science Vector Construction
    snv_m = snv(m)
    sg_m = apply_sg_derivative(snv_m)

    # Combine into the 104-dim feature vector
    vec = np.concatenate([m, snv_m, sg_m, s]).reshape(1, -1)
    scaled = scaler.transform(vec)

    cls = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    labels = {0: "Fresh", 1: "Fungicide", 2: "Insecticide"}
    return labels.get(cls, "Unknown"), float(prob[cls])

def predict_pesticide_hybrid(cube: np.ndarray, rgb_u8: np.ndarray, model_data: dict, mask=None):
    model = model_data.get("model", model_data.get("model_s1"))
    scaler = model_data["scaler"]

    # Extract RGB features for hybrid model
    img_f = rgb_u8.astype(np.float32) / 255.0
    if mask is not None and mask.any():
        rgb_px = img_f[mask]
        rm, rs = np.mean(rgb_px, axis=0), np.std(rgb_px, axis=0)
    else:
        rm, rs = np.mean(img_f, axis=(0, 1)), np.std(img_f, axis=(0, 1))

    rgb_feats = np.concatenate([rm, rs])

    # Extract Spectral features
    if mask is not None and mask.any():
        px = cube[mask]
        m, s = px.mean(axis=0), px.std(axis=0)
    else:
        m, s = cube.mean(axis=(0, 1)), cube.std(axis=(0, 1))

    snv_m = snv(m)
    sg_m = apply_sg_derivative(snv_m)
    ratios = compute_spectral_ratios(m)

    # Final Hybrid Vector
    vec = np.concatenate([snv_m, sg_m, s, ratios]).reshape(1, -1)
    scaled = scaler.transform(vec)

    cls = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]

    labels = {0: "Fresh", 1: "Pesticide"} if len(probs) == 2 else {0: "Fresh", 1: "Fungicide", 2: "Insecticide"}
    prob_map = {labels[i]: float(probs[i]) for i in range(len(probs))}

    return prob_map, labels[cls], float(probs[cls]), scaled

def get_zonal_features(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zonal analysis: splits fruit into Center, Mid, and Edge regions.
    Returns a concatenated vector of 3 * 104 = 312 features.
    """
    def _extract(c, m):
        if not m.any(): return np.zeros(104, dtype=np.float32)
        px = c[m]
        m_s, std_s = px.mean(axis=0), px.std(axis=0)
        snv_m = snv(m_s)
        sg_m = apply_sg_derivative(snv_m)
        ratios = compute_spectral_ratios(m_s)
        return np.concatenate([np.concatenate([snv_m, sg_m, std_s]), ratios])

    h, w = mask.shape
    cy, cx = h // 2, w // 2

    # Center (30%)
    dy, dx = int(h * 0.15), int(w * 0.15)
    c_mask = np.zeros_like(mask, dtype=bool)
    c_mask[cy-dy:cy+dy, cx-dx:cx+dx] = True
    c_mask &= mask

    # Mid (30-60%)
    m_mask = np.zeros_like(mask, dtype=bool)
    dy_m, dx_m = int(h * 0.3), int(w * 0.3)
    m_mask[cy-dy_m:cy+dy_m, cx-dx_m:cx+dx_m] = True
    m_mask &= mask
    m_mask &= ~c_mask

    # Edge (Remaining)
    e_mask = mask & ~m_mask & ~c_mask

    return np.concatenate([_extract(cube, c_mask), _extract(cube, m_mask), _extract(cube, e_mask)])

def get_shap_fingerprint(feat_scaled, model_data):
    # Use SHAP to explain which bands caused the prediction
    model = model_data["model_s1"]
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(feat_scaled)

    # Handle binary vs multi-class SHAP output
    res = vals[1][0] if isinstance(vals, list) else vals[0]

    f_names = [f"SNV_{i+1}" for i in range(31)] + \
              [f"SG_{i+1}" for i in range(30)] + \
              [f"STD_{i+1}" for i in range(31)] + \
              [f"Ratio_{i+1}" for i in range(12)]

    return [(name, float(v)) for name, v in zip(f_names, res)]

def resolve_pipeline_ckpt(pipeline: str) -> Path:
    mapping = {
        "A": PROJECT_DIR / "models/checkpoints/run_trained/net_best.pth",
        "B": PROJECT_DIR / "models/checkpoints/run_trained/net_best.pth",
        "C": PROJECT_DIR / "models/checkpoints/run_apple_spectrofood_finetuned/net_best.pth",
        "fallback": PROJECT_DIR / "models/checkpoints/run_trained/net_best.pth",
    }
    p = mapping.get(pipeline, mapping["fallback"])
    if not p.is_file():
        raise FileNotFoundError(f"Pipeline {pipeline} weights missing at {p}")
    return p

def route_and_infer(img_path: str | Path, model=None, device=None, skip=False):
    from routing import route_image, RoutingDecision
    dec = route_image(img_path)

    if dec.tier == 3 or dec.pipeline_to_run is None or skip:
        return dec, None

    ckpt = resolve_pipeline_ckpt(dec.pipeline_to_run)
    if model is None or device is None:
        model, device = load_model(ckpt, device)

    return dec, run_model1(img_path, model, device)

def list_test_pairs() -> list[str]:
    # Search standard test layouts
    layouts = (
        (PROJECT_DIR / "data" / "raw" / "Test_RGB", PROJECT_DIR / "data" / "raw" / "Test_Spec"),
        (PROJECT_DIR / "data" / "raw" / "Agro-HSR" / "Test_RGB", PROJECT_DIR / "data" / "raw" / "Agro-HSR" / "Test_Spec"),
    )
    for rgb_d, spec_d in layouts:
        if rgb_d.is_dir() and spec_d.is_dir():
            return [Path(p).stem for p in glob.glob(str(rgb_d / "*.jpg"))
                    if (spec_d / f"{Path(p).stem}.mat").is_file()]
    return []

def load_gt_hsi(stem: str) -> np.ndarray:
    import hdf5storage
    # Defaulting to a standard test path for consistency
    path = PROJECT_DIR / "data" / "raw" / "Test_RGB" / f"{stem}.mat"
    return hdf5storage.loadmat(str(path))["cube"].astype(np.float32)

def crop_gt(pred, gt):
    hp, wp = pred.shape[:2]
    hg, wg = gt.shape[:2]
    if (hp, wp) == (hg, wg): return gt
    # Center crop
    top, left = max(0, (hg-hp)//2), max(0, (wg-wp)//2)
    return gt[top:top+hp, left:left+wp, :]

def compute_metrics(pred, gt, eps=1e-3):
    p, g = pred.astype(np.float32), gt.astype(np.float32)
    diff = p - g
    mrae = float(np.mean(np.abs(diff) / (np.abs(g) + eps)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    p_lo, p_hi = p.min(), p.max()
    if p_hi - p_lo < 1e-8: psnr = 0.0
    else:
        p_rescaled = (p - p_lo) / (p_hi - p_lo)
        g_rescaled = np.clip(g, 0.0, 1.0)
        mse = np.mean((p_rescaled - g_rescaled)**2)
        psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else 100.0
    return {"mrae": mrae, "rmse": rmse, "psnr": psnr}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default="output_cube.npy")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, d = load_model(args.ckpt, dev)
    res = run_model1(args.rgb, m, d)
    np.save(args.out, res)
