import os
import json
import pandas as pd
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent

# 1. Define all manifests to check for used images
manifests = [
    ROOT / "experiments/clean_apple_pesticide_manifest.csv",
    ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_extended_blind.json",
    ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_balanced_blind.json",
    ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_blind_50.json",
    ROOT / "data/pesticides/vaishnavi_2023/manifest.json",
    ROOT / "data/pesticides/vaishnavi_2023/rgb/manifest.json",
]

used_images = set()

for m_path in manifests:
    if not m_path.exists():
        continue

    try:
        if m_path.suffix == '.csv':
            df = pd.read_csv(m_path)
            if 'path' in df.columns:
                for p in df['path']:
                    used_images.add(Path(p).name)
        elif m_path.suffix == '.json':
            with open(m_path, 'r') as f:
                data = json.load(f)
                # Handle different JSON structures
                if 'pairs' in data:
                    for pair in data['pairs'].values():
                        if 'rgb_path' in pair: used_images.add(Path(pair['rgb_path']).name)
                        if 'bil_path' in pair: used_images.add(Path(pair['bil_path']).name)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for val in item.values():
                                if isinstance(val, str) and (val.endswith('.png') or val.endswith('.jpg')):
                                    used_images.add(Path(val).name)
                elif isinstance(data, dict):
                    for val in data.values():
                        if isinstance(val, str) and (val.endswith('.png') or val.endswith('.jpg')):
                            used_images.add(Path(val).name)
    except Exception as e:
        print(f"Error reading manifest {m_path}: {e}")

print(f"Found {len(used_images)} unique images mentioned in manifests.")

# 2. Find all images on disk in data directories
data_dirs = [
    ROOT / "data/pesticides/vaishnavi_2023/rgb",
    ROOT / "data/pesticides/vaishnavi_2023/Apple_samples",
    ROOT / "data/pesticides/vaishnavi_2023/Fungicide_Apple",
    ROOT / "data/pesticides/vaishnavi_2023/Pesticide_Apple",
]

all_images = []
for d in data_dirs:
    if not d.is_dir(): continue
    for path in d.rglob("*"):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bil"):
            all_images.append(path)

print(f"Found {len(all_images)} total image/spectral files on disk.")

# 3. Find unseen images
unseen_images = []
for img in all_images:
    if img.name not in used_images:
        unseen_images.append(img)

print(f"Found {len(unseen_images)} unseen images.")

# 4. Create The Vault
vault_dir = ROOT / "data/The_Vault"
vault_dir.mkdir(parents=True, exist_ok=True)

count = 0
for img in unseen_images:
    # Move to vault while preserving some structure if needed, but here we'll just copy to avoid breaking existing paths
    # Copying is safer than moving if manifests are fragile.
    try:
        shutil.copy2(img, vault_dir / img.name)
        count += 1
    except Exception as e:
        print(f"Error copying {img.name}: {e}")

print(f"Successfully copied {count} unseen images to {vault_dir}")
