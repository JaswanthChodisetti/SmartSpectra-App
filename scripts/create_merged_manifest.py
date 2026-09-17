import json
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
VAISHNAVI_BLIND = ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_true_blind.json"
FRUITVISION_ROOT = ROOT / "data/Fruit_Vision/extracted/Fruits Original/Apple"
MERGED_MANIFEST_PATH = ROOT / "data/pesticides/apple_merged_blind_manifest.json"

def generate_merged_manifest():
    merged_pairs = {}

    # 1. Load Original Vaishnavi Blind Pairs
    if VAISHNAVI_BLIND.exists():
        with open(VAISHNAVI_BLIND, "r") as f:
            vaishnavi_data = json.load(f)
            merged_pairs.update(vaishnavi_data.get("pairs", {}))
        print(f"Loaded {len(vaishnavi_data.get('pairs', {}))} samples from Vaishnavi blind set.")
    else:
        print("Vaishnavi blind manifest not found. Starting fresh.")

    # 2. Add FruitVision Apples
    # Mapping: Fresh -> Fresh, Formalin-mixed -> Pesticide, Rotten -> Pesticide
    label_map = {
        "Fresh": "Fresh",
        "Formalin-mixed": "Pesticide",
        "Rotten": "Pesticide"
    }

    fruitvision_count = 0
    for folder_name, mapped_label in label_map.items():
        folder_path = FRUITVISION_ROOT / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_name} not found at {folder_path}")
            continue

        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
        for i, img_path in enumerate(images):
            # Create a unique ID for each FruitVision image
            pair_id = f"fv_{folder_name}_{i:04d}"
            merged_pairs[pair_id] = {
                "rgb_path": str(img_path),
                "label": mapped_label
            }
            fruitvision_count += 1

    print(f"Added {fruitvision_count} samples from FruitVision dataset.")

    # 3. Save the merged manifest
    manifest = {"pairs": merged_pairs}
    with open(MERGED_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully created merged manifest at {MERGED_MANIFEST_PATH}")
    print(f"Total samples in merged blind set: {len(merged_pairs)}")

if __name__ == "__main__":
    generate_merged_manifest()
