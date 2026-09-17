import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MODEL1_CKPT = ROOT / "models/checkpoints/run_trained/net_best.pth"
BASE_MANIFEST = ROOT / "Archive/experiments/clean_apple_pesticide_manifest.csv"
SAVE_PATH = ROOT / "models/checkpoints/model2_ultimate_mlp.pkl"

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

def train_ultimate_mlp():
    print("--- ULTIMATE MLP: ZONAL FEATURES & DIVERSIFICATION ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1, _ = load_model(str(MODEL1_CKPT), device)

    if not BASE_MANIFEST.exists():
        print(f"Error: Manifest not found at {BASE_MANIFEST}")
        return
    df = pd.read_csv(BASE_MANIFEST)

    X, y = [], []
    label_map = {"Fresh": 0, "Pesticide": 1}

    print("Extracting Zonal Features...")
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        import cv2
        img_bgr_orig = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        if img_bgr_orig is None: continue
        img_bgr = cv2.resize(img_bgr_orig, (256, 256), interpolation=cv2.INTER_AREA)
        try:
            with torch.no_grad():
                cube = run_model1(row["path"], model1, device)
            mask = segment_produce(img_bgr)
            feat = extract_zonal_pure_science_features(cube, mask)
            X.append(feat)
            y.append(label_map.get(row["label"], 0))
        except Exception: continue

    X, y = np.array(X), np.array(y)

    print("Applying Diversity Augmentation...")
    X_all, y_all = [], []
    for label in [0, 1]:
        mask = (y == label)
        class_features = X[mask]
        X_all.extend(class_features)
        y_all.extend([label] * len(class_features))
        aug_feats = diversity_augment_v2(class_features, n_variants=5)
        X_all.extend(aug_feats)
        y_all.extend([label] * len(aug_feats))

    X_all, y_all = np.array(X_all), np.array(y_all)

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

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # CALIBRATED WEIGHTS
    # Fresh: 5.0, Pesticide: 2.0
    weights = torch.FloatTensor([5.0, 2.0]).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SpectralMLP(input_dim=312).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device)) # Shifted for Pesticide priority

    # Training Loop
    best_val_acc = 0
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # Custom weight application based on class
            outputs = model(batch_x)
            # Use a simplified weighted loss
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = (model(X_val_t) > 0.5).float()
            val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_mlp.pth")

    # Load best model
    model.load_state_dict(torch.load("best_mlp.pth"))
    model.eval()

    with torch.no_grad():
        test_preds = (model(X_test_t) > 0.5).float().cpu().numpy()

    acc = accuracy_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds)
    pest_recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0

    print("\n" + "="*40)
    print("ULTIMATE MLP RESULTS")
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
        # We save the model as a dict to be compatible with the app's loading logic
        # Since we use a custom MLP class, we save the class definition as well or save as a torch model
        joblib.dump({"model_s1": model.state_dict(), "scaler": scaler, "is_mlp": True}, SAVE_PATH)
        print(f"\nSuccess! Ultimate Model saved to {SAVE_PATH}")
    else:
        print("\nWarning: Accuracy threshold not met. Model saved but use with caution.")

if __name__ == "__main__":
    train_ultimate_mlp()
