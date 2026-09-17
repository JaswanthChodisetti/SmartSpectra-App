import os
import json
import numpy as np
import torch
import joblib
import cv2
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"
MODEL2_CKPT = ROOT / "models/checkpoints/model2_ultimate_mlp.pkl"

# Import from demo and scripts
sys.path.insert(0, str(ROOT))
from scripts.train_ultimate_mlp import SpectralMLP
from inference import run_model1, load_model, snv, apply_sg_derivative, compute_spectral_ratios, segment_produce

def extract_pure_science_features(cube, img_bgr, model_data, mask=None):
    scaler = model_data["scaler"]
    # Use the new zonal feature extraction from inference.py
    from inference import extract_zonal_pure_science_features
    features = extract_zonal_pure_science_features(cube, mask)
    return scaler.transform(features.reshape(1, -1))[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python blind_test_manifest.py <path_to_manifest_json>")
        return

    manifest_path = Path(sys.argv[1])
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Manifest-Based Blind Test on {device}...")

    # Load Models
    model1, _ = load_model(str(MODEL1_CKPT), device)
    model_data = joblib.load(str(MODEL2_CKPT))

    if model_data.get("is_mlp", False):
        import torch.nn as nn
        from scripts.train_ultimate_mlp import SpectralMLP
        model2 = SpectralMLP().to(device)
        model2.load_state_dict(model_data["model_s1"])
        model2.eval()
    else:
        model2 = model_data["model_s1"]


    # Load Manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pairs = manifest["pairs"]
    print(f"Found {len(pairs)} samples in manifest. Starting inference...")

    y_true, y_pred = [], []

    for pair_id, info in tqdm(pairs.items()):
        rgb_path = Path(info["rgb_path"])
        label_str = info["label"].lower()

        if not rgb_path.exists():
            continue

        true_label = 0 if label_str == "fresh" else 1

        try:
            img_bgr = cv2.imread(str(rgb_path))
            if img_bgr is None: continue
            img_256 = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

            with torch.no_grad():
                cube = run_model1(str(rgb_path), model1, device)

            mask = segment_produce(img_256)
            feat = extract_pure_science_features(cube, img_bgr, model_data, mask=mask)

            # Prediction based on model type
            if model_data.get("is_mlp", False):
                with torch.no_grad():
                    feat_t = torch.FloatTensor(feat).unsqueeze(0).to(device)
                    pred = (model2(feat_t) > 0.5).int().item()
            else:
                pred = model2.predict(feat.reshape(1, -1))[0]


            y_true.append(true_label)
            y_pred.append(pred)

        except Exception as e:
            print(f"Error processing {pair_id}: {e}")

    if not y_true:
        print("No valid samples were processed.")
        return

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"MANIFEST BLIND TEST RESULTS: {manifest_path.name}")
    print("="*40)
    print(f"Total Samples Processed: {len(y_true)}")
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Fresh", "Pesticide"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("="*40)

if __name__ == "__main__":
    main()
