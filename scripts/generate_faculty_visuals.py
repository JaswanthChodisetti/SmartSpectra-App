import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models/checkpoints/model2_ultimate_mlp.pkl"
MANIFEST_PATH = ROOT / "data/pesticides/apple_final_blind_manifest.json"
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"
PLOT_DIR = ROOT / "demo/assets/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Import from demo
import sys
sys.path.insert(0, str(ROOT / "demo"))
from inference import run_model1, load_model, extract_zonal_pure_science_features, segment_produce

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

def get_blind_predictions():
    print("--- PREPARING DATA FOR VISUALS ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1, _ = load_model(str(MODEL1_CKPT), device)
    ckpt = joblib.load(MODEL_PATH)
    scaler = ckpt["scaler"]
    mlp_model = SpectralMLP(input_dim=312).to(device)
    mlp_model.load_state_dict(ckpt["model_s1"])
    mlp_model.eval()

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]

    # For faculty visuals, we use a representative subset to keep computation time reasonable
    # while maintaining statistical significance.
    if len(pairs) > 100:
        print(f"Reducing manifest from {len(pairs)} to 100 representative samples...")
        import random
        sampled_pids = random.sample(list(pairs.keys()), 100)
        pairs = {pid: pairs[pid] for pid in sampled_pids}

    X_scaled, y_true, probs = [], [], []

    for pid, info in tqdm(pairs.items()):
        import cv2
        img_path = info["rgb_path"]
        img_bgr_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr_orig is None: continue
        img_bgr = cv2.resize(img_bgr_orig, (256, 256), interpolation=cv2.INTER_AREA)
        try:
            with torch.no_grad():
                cube = run_model1(img_path, model1, device)
            mask = segment_produce(img_bgr)
            feat = extract_zonal_pure_science_features(cube, mask)
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            X_scaled.append(feat_scaled[0])

            feat_t = torch.FloatTensor(feat_scaled).to(device)
            prob = mlp_model(feat_t).item()
            probs.append(prob)
            y_true.append(1 if info["label"] == "Pesticide" else 0)
        except Exception:
            continue

    return np.array(X_scaled), np.array(y_true), np.array(probs), scaler

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fresh', 'Pesticide'],
                yticklabels=['Fresh', 'Pesticide'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def generate_shap_plot(X, y, mlp_model, scaler):
    print("Generating SHAP analysis (this may take a few minutes)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrapper for SHAP to work with PyTorch
    def model_predict(data):
        data_t = torch.FloatTensor(data).to(device)
        with torch.no_grad():
            return mlp_model(data_t).cpu().numpy()

    # Use a subset for SHAP to be faster
    subset_idx = np.random.choice(len(X), min(200, len(X)), replace=False)
    X_subset = X[subset_idx]

    # KernelExplainer is model-agnostic
    explainer = shap.KernelExplainer(model_predict, shap.sample(X, 50))
    shap_values = explainer.shap_values(X_subset)

    # Handle SHAP output for binary classification
    # If shap_values is a list (common in older SHAP/certain models), it contains values for each class.
    if isinstance(shap_values, list):
        # For binary, we only care about the positive class (index 1)
        if len(shap_values) > 1:
            shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    plt.figure(figsize=(10, 6))
    # Calculate mean absolute impact per feature across the subset
    mean_shap = np.abs(shap_values).mean(axis=0)
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.flatten()

    top_indices = np.argsort(mean_shap)[-20:]


    # Ensure we pass lists to barh to avoid numpy scalar conversion issues in some Matplotlib versions
    y_pos = np.arange(20).astype(float).tolist()
    widths = mean_shap[top_indices].flatten().astype(float).tolist()

    plt.barh(y_pos, widths, color='skyblue')
    plt.yticks(y_pos, [f"Feat {i}" for i in top_indices])
    plt.xlabel("Mean |SHAP Value| (Impact on Prediction)")
    plt.title("Top 20 Pure Science Features Driving Pesticide Detection")
    plt.savefig(PLOT_DIR / "shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: shap_summary.png")

def main():
    X_scaled, y_true, probs, scaler = get_blind_predictions()

    # 1. Original Results (T=0.5)
    preds_orig = (probs >= 0.5).astype(int)
    plot_confusion_matrix(y_true, preds_orig, "Blind Test: Original (T=0.5)", "cm_original.png")

    # 2. Calibrated Results (T=0.2)
    preds_calib = (probs >= 0.2).astype(int)
    plot_confusion_matrix(y_true, preds_calib, "Blind Test: Calibrated (T=0.2)", "cm_calibrated.png")

    # 3. SHAP Analysis
    # We need the model in the right state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = joblib.load(MODEL_PATH)
    mlp_model = SpectralMLP(input_dim=312).to(device)
    mlp_model.load_state_dict(ckpt["model_s1"])
    mlp_model.eval()
    generate_shap_plot(X_scaled, y_true, mlp_model, scaler)

    print("\nAll faculty visuals generated in demo/assets/plots/")

if __name__ == "__main__":
    main()
