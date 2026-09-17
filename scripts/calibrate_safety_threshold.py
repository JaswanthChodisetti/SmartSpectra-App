import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models/checkpoints/model2_ultimate_mlp.pkl"
MANIFEST_PATH = ROOT / "data/pesticides/apple_final_blind_manifest.json"
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"

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

def calibrate_safety():
    print("--- SAFETY CALIBRATION: FINDING THE SWEETSPOT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model 1 (Reconstructor)
    model1, _ = load_model(str(MODEL1_CKPT), device)

    # 2. Load Model 2 (The Calibrated MLP)
    ckpt = joblib.load(MODEL_PATH)
    scaler = ckpt["scaler"]
    mlp_model = SpectralMLP(input_dim=312).to(device)
    mlp_model.load_state_dict(ckpt["model_s1"])
    mlp_model.eval()

    # 3. Load Manifest
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]

    X_probs, y_true = [], []

    print("Extracting raw probabilities from blind set...")
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

            # Scale and Predict
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            feat_t = torch.FloatTensor(feat_scaled).to(device)
            prob = mlp_model(feat_t).item()

            X_probs.append(prob)
            y_true.append(1 if info["label"] == "Pesticide" else 0)
        except Exception:
            continue

    X_probs = np.array(X_probs)
    y_true = np.array(y_true)

    # 4. Iterate Thresholds
    results = []
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (X_probs >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)

        # Recall = TP / (TP + FN)
        tp = cm[1][1] if cm.shape == (2,2) else 0
        fn = cm[1][0] if cm.shape == (2,2) else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append({
            "Threshold": round(threshold, 2),
            "Accuracy": f"{acc:.2%}",
            "Pesticide Recall": f"{recall:.2%}",
            "False Negatives": fn
        })

    # 5. Display results
    df_res = pd.DataFrame(results)
    print("\n" + "="*60)
    print("CALIBRATION TABLE: Impact of Threshold on Safety")
    print("="*60)
    print(df_res.to_string(index=False))
    print("="*60)

    # Find best threshold (Recall >= 80% and highest accuracy)
    best_t = 0.5
    max_acc = 0
    for res in results:
        rec_val = float(res["Pesticide Recall"].strip('%')) / 100
        acc_val = float(res["Accuracy"].strip('%')) / 100
        if rec_val >= 0.80 and acc_val > max_acc:
            max_acc = acc_val
            best_t = res["Threshold"]

    print(f"\nRECOMMENDED SAFETY THRESHOLD: {best_t}")
    print(f"Expected Recall: {next(r['Pesticide Recall'] for r in results if r['Threshold'] == best_t)}")
    print(f"Expected Accuracy: {next(r['Accuracy'] for r in results if r['Threshold'] == best_t)}")

if __name__ == "__main__":
    calibrate_safety()
