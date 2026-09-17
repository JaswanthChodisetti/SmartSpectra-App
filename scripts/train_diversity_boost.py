import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"
BASE_MANIFEST = ROOT / "Archive/experiments/clean_apple_pesticide_manifest.csv"
SAVE_PATH = ROOT / "models/checkpoints/model2_diversity_boost.pkl"

# Import from demo
import sys
sys.path.insert(0, str(ROOT / "demo"))
from inference import run_model1, load_model, snv, apply_sg_derivative, segment_produce, compute_spectral_ratios

def extract_pure_science_features(image_path, model1, device):
    import cv2
    img_bgr_orig = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr_orig is None: return None
    img_bgr = cv2.resize(img_bgr_orig, (256, 256), interpolation=cv2.INTER_AREA)
    try:
        with torch.no_grad():
            cube = run_model1(image_path, model1, device)
    except Exception: return None
    mask = segment_produce(img_bgr)
    if mask.any():
        px = cube[mask]
        mean_spec, std_spec = px.mean(axis=0), px.std(axis=0)
    else:
        mean_spec, std_spec = cube.mean(axis=(0, 1)), cube.std(axis=(0, 1))

    snv_mean = snv(mean_spec)
    sg_mean = apply_sg_derivative(snv_mean)
    ratio_features = compute_spectral_ratios(mean_spec)
    return np.concatenate([np.concatenate([snv_mean, sg_mean, std_spec]), ratio_features])

def diversity_augment_v2(class_features, n_variants=10):
    variants = []
    num_samples = len(class_features)
    feat_dim = class_features.shape[1]

    for _ in range(n_variants * num_samples):
        base_feat = class_features[np.random.randint(0, num_samples)]
        mode = np.random.choice(['shift', 'mixup', 'offset', 'warp'])

        if mode == 'shift':
            shift = np.random.uniform(-0.003, 0.003, size=feat_dim)
            variants.append(base_feat + shift)
        elif mode == 'mixup':
            f2 = class_features[np.random.randint(0, num_samples)]
            alpha = np.random.uniform(0.3, 0.7)
            variants.append(alpha * base_feat + (1 - alpha) * f2)
        elif mode == 'offset':
            offset = np.random.uniform(-0.002, 0.002)
            variants.append(base_feat + offset)
        elif mode == 'warp':
            warp = np.linspace(np.random.uniform(0.98, 1.02),
                               np.random.uniform(0.98, 1.02), feat_dim)
            variants.append(base_feat * warp)

    return np.array(variants)

def train_diversity_boost():
    print("--- BASELINE DIVERSIFICATION: FINAL SAFE RUN ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1, _ = load_model(str(MODEL1_CKPT), device)

    if not BASE_MANIFEST.exists():
        print(f"Error: Manifest not found at {BASE_MANIFEST}")
        return
    df = pd.read_csv(BASE_MANIFEST)

    X, y = [], []
    label_map = {"Fresh": 0, "Pesticide": 1}

    print("Extracting features...")
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        feat = extract_pure_science_features(row["path"], model1, device)
        if feat is None: continue
        X.append(feat)
        y.append(label_map.get(row["label"], 0))

    X = np.array(X)
    y = np.array(y)

    print("Applying Baseline Diversification Augmentation...")
    X_all, y_all = [], []
    for label in [0, 1]:
        mask = (y == label)
        class_features = X[mask]
        X_all.extend(class_features)
        y_all.extend([label] * len(class_features))
        aug_feats = diversity_augment_v2(class_features, n_variants=5)
        X_all.extend(aug_feats)
        y_all.extend([label] * len(aug_feats))

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    base_weights = compute_sample_weight('balanced', y_train)
    final_weights = np.where(y_train == 0, base_weights * 5.0, base_weights * 2.0)

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=3,
        gamma=0.8,
        reg_alpha=0.3,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30
    )

    model.fit(X_train_scaled, y_train, sample_weight=final_weights, eval_set=[(X_val_scaled, y_val)], verbose=True)

    test_preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds)
    pest_recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0

    print("\n" + "="*40)
    print("BASELINE DIVERSIFICATION RESULTS")
    print("="*40)
    print(f"Overall Accuracy: {acc:.2%}")
    print(f"Pesticide Recall (Safety): {pest_recall:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))
    print("Confusion Matrix:\n", cm)
    print("="*40)

    if pest_recall < 0.80:
        print("\n❌ SAFETY SWITCH TRIGGERED: Pesticide Recall fell below 80%. Model discarded.")
        return

    if acc > 0.80:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_s1": model, "scaler": scaler}, SAVE_PATH)
        print(f"\nSuccess! Model saved to {SAVE_PATH}")
    else:
        print("\nWarning: Accuracy threshold not met. Model saved but use with caution.")

if __name__ == "__main__":
    train_diversity_boost()
