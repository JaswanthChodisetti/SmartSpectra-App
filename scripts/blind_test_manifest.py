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

# Project Root setup for portable execution
ROOT = Path(__file__).resolve().parent.parent
# These are our gold-standard weights
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"
MODEL2_CKPT = ROOT / "models/checkpoints/model2_ultimate_mlp.pkl"

# Add ROOT to path so we can grab the inference logic
sys.path.insert(0, str(ROOT))
from scripts.train_ultimate_mlp import SpectralMLP
from inference import run_model1, load_model, segment_produce

def extract_features(cube, img_bgr, model_data, mask=None):
    # Get the scaler from the loaded model pkl
    scaler = model_data["scaler"]

    # We use the zonal approach to get a more robust average across the fruit
    from inference import extract_zonal_pure_science_features
    feat_vec = extract_zonal_pure_science_features(cube, mask)

    # Transform the raw vector into the scaled space the model expects
    return scaler.transform(feat_vec.reshape(1, -1))[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python blind_test_manifest.py <manifest.json>")
        return

    manifest_path = Path(sys.argv[1])
    if not manifest_path.exists():
        print(f"Error: Could not find manifest at {manifest_path}")
        return

    # Setup device - fallback to CPU if GPU isn't available
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running blind test evaluation on {dev}...")

    # Load the reconstruction net (Model 1) and the classifier (Model 2)
    model1, _ = load_model(str(MODEL1_CKPT), dev)
    model_data = joblib.load(str(MODEL2_CKPT))

    # Initialize the MLP architecture for the classifier
    if model_data.get("is_mlp", False):
        model2 = SpectralMLP().to(dev)
        model2.load_state_dict(model_data["model_s1"])
        model2.eval()
    else:
        # Fallback to XGBoost if the pkl is not an MLP
        model2 = model_data["model_s1"]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pairs = manifest["pairs"]
    print(f"Evaluating {len(pairs)} pairs. Starting loop...")

    y_true, y_pred = [], []

    for p_id, info in tqdm(pairs.items()):
        rgb_p = Path(info["rgb_path"])
        # Convert label to binary: Fresh=0, Pesticide=1
        label_str = info["label"].lower()

        if not rgb_p.exists():
            continue

        # Binary target for the final report
        true_val = 0 if label_str == "fresh" else 1

        try:
            # Get the image and resize to MST++ standard (256x256)
            img = cv2.imread(str(rgb_p))
            if img is None: continue
            img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

            with torch.no_grad():
                # Step 1: RGB -> HSI Reconstruction
                cube = run_model1(str(rgb_p), model1, dev)

            # Step 2: Masking and Feature Extraction
            mask = segment_produce(img_256)
            feat = extract_features(cube, img, model_data, mask=mask)

            # Step 3: Final Prediction
            if model_data.get("is_mlp", False):
                with torch.no_grad():
                    t_feat = torch.FloatTensor(feat).unsqueeze(0).to(dev)
                    pred = (model2(t_feat) > 0.5).int().item()
            else:
                # Use XGBoost predict
                pred = model2.predict(feat.reshape(1, -1))[0]

            y_true.append(true_val)
            y_pred.append(pred)

        except Exception as e:
            print(f"Skipping {p_id} due to error: {e}")

    if not y_true:
        print("No valid samples processed. Check your manifest paths.")
        return

    # Final scientific metrics
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"RESULTS: {manifest_path.name}")
    print("="*40)
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.2%}")
    print("\nDetailed Report:\n", classification_report(y_true, y_pred, target_names=["Fresh", "Pesticide"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("="*40)

if __name__ == "__main__":
    main()
