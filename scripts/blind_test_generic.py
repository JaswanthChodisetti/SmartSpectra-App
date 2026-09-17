import os
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
MODEL2_CKPT = ROOT / "models/checkpoints/model2_final.pkl"

# Import from demo
import sys
sys.path.insert(0, str(ROOT / "demo"))
from inference import run_model1, load_model, snv, apply_sg_derivative, compute_spectral_ratios, segment_produce

def extract_pure_science_features(cube, img_bgr, model_data, mask=None):
    scaler = model_data["scaler"]
    if mask is not None and mask.any():
        px = cube[mask]
        mean_spec, std_spec = px.mean(axis=0), px.std(axis=0)
    else:
        mean_spec, std_spec = cube.mean(axis=(0, 1)), cube.std(axis=(0, 1))

    snv_mean = snv(mean_spec)
    sg_mean = apply_sg_derivative(snv_mean)
    ratio_features = compute_spectral_ratios(mean_spec)

    features = np.concatenate([
        np.concatenate([snv_mean, sg_mean, std_spec]),
        ratio_features
    ])
    return scaler.transform(features.reshape(1, -1))[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python blind_test_generic.py <path_to_image_folder>")
        return

    image_dir = Path(sys.argv[1])
    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Generic Blind Test on {device}...")

    # Load Models
    model1, _ = load_model(str(MODEL1_CKPT), device)
    model_data = joblib.load(str(MODEL2_CKPT))
    model2 = model_data["model_s1"]

    # Find Images
    valid_exts = (".jpg", ".jpeg", ".png")
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts]

    if not images:
        print("No valid images found in the folder.")
        return

    print(f"Found {len(images)} samples. Starting inference...")

    y_true, y_pred = [], []
    label_map = {"fresh": 0, "pesticide": 1}

    for img_path in tqdm(images):
        # 1. Determine True Label from Filename
        fname = img_path.name.lower()
        if "fresh" in fname:
            true_label = 0
        elif "pesticide" in fname or "fungicide" in fname or "insecticide" in fname:
            true_label = 1
        else:
            continue # Skip if label is unclear

        try:
            # 2. Preprocess & Reconstruct Cube
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            img_256 = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

            with torch.no_grad():
                cube = run_model1(str(img_path), model1, device)

            # 3. Segment & Extract Features
            mask = segment_produce(img_256)
            feat = extract_pure_science_features(cube, img_bgr, model_data, mask=mask)

            # 4. Predict
            pred = model2.predict(feat.reshape(1, -1))[0]

            y_true.append(true_label)
            y_pred.append(pred)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    if not y_true:
        print("No valid labeled samples were processed.")
        return

    # Final Evaluation
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print("GENERIC BLIND TEST RESULTS")
    print("="*40)
    print(f"Total Samples Processed: {len(y_true)}")
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Fresh", "Pesticide"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("="*40)

if __name__ == "__main__":
    main()
