import os
import json
import numpy as np
import torch
import joblib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_extended_blind.json"
MODEL_FINAL = ROOT / "models/checkpoints/model2_final.pkl"
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"
PLOT_DIR = ROOT / "demo/assets/plots"

import sys
sys.path.insert(0, str(ROOT / "demo"))
from inference import run_model1, load_model, snv, apply_sg_derivative, compute_spectral_ratios, segment_produce

def load_bil_cube(path):
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data.reshape((256, 256, 31))
    except Exception:
        try:
            data = np.fromfile(path, dtype=np.int16)
            return data.reshape((256, 256, 31)).astype(np.float32)
        except Exception:
            return None

def extract_features(cube, img_bgr, model_data, mask=None):
    scaler = model_data["scaler"]
    if mask is not None and mask.any():
        px = cube[mask]
        mean_spec, std_spec = px.mean(axis=0), px.std(axis=0)
    else:
        mean_spec, std_spec = cube.mean(axis=(0, 1)), cube.std(axis=(0, 1))
    snv_mean = snv(mean_spec)
    sg_mean = apply_sg_derivative(snv_mean)
    ratio_features = compute_spectral_ratios(mean_spec)
    features = np.concatenate([np.concatenate([snv_mean, sg_mean, std_spec]), ratio_features])
    return scaler.transform(features.reshape(1, -1))[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1, _ = load_model(str(MODEL1_CKPT), device)
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]
    
    model_data = joblib.load(str(MODEL_FINAL))
    model = model_data.get("model_s1", model_data.get("model"))
    
    y_true, y_pred = [], []
    label_map = {"fresh": 0, "fungicide": 1, "insecticide": 1}
    
    for name, info in tqdm(pairs.items(), desc="Generating Sweetspot CM"):
        rgb_path = info.get("rgb_path", "MISSING")
        bil_path = info.get("bil_path", "MISSING")
        true_label = label_map.get(info["label"], 0)
        try:
            cube = None
            img_bgr = None
            if rgb_path != "MISSING":
                img_bgr = cv2.imread(rgb_path)
                if img_bgr is not None:
                    with torch.no_grad():
                        cube = run_model1(rgb_path, model1, device)
            elif bil_path != "MISSING":
                cube = load_bil_cube(bil_path)
                img_bgr = np.full((256, 256, 3), 128, dtype=np.uint8)
            if cube is None: continue
            mask = None
            if img_bgr is not None:
                img_256 = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
                mask = segment_produce(img_256)
            feat = extract_features(cube, img_bgr, model_data, mask=mask)
            pred = model.predict(feat.reshape(1, -1))[0]
            y_true.append(true_label)
            y_pred.append(pred)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fresh', 'Pesticide'], yticklabels=['Fresh', 'Pesticide'])
    plt.title("Sweetspot Model Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    PLOT_DIR = ROOT / "demo/assets/plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / "cm_sweetspot.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Successfully saved cm_sweetspot.png")

if __name__ == "__main__":
    main()
