import json
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
VAISHNAVI_BLIND = ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_true_blind.json"
FRUITVISION_ROOT = ROOT / "data/Fruit_Vision/extracted/Fruits Original/Apple"
REFINED_MANIFEST_PATH = ROOT / "data/pesticides/apple_refined_blind_manifest.json"

def generate_refined_manifest():
    refined_pairs = {}

    # 1. Load Original Vaishnavi Blind Pairs (The 18 images)
    if VAISHNAVI_BLIND.exists():
        with open(VAISHNAVI_BLIND, "r") as f:
            vaishnavi_data = json.load(f)
            refined_pairs.update(vaishnavi_data.get("pairs", {}))
        print(f"Loaded {len(vaishnavi_data.get('pairs', {}))} samples from Vaishnavi blind set.")
    else:
        print("Vaishnavi blind manifest not found.")

    # 2. Add ONLY Fresh and Formalin-mixed from FruitVision
    # Mapping: Fresh -> Fresh, Formalin-mixed -> Pesticide
    label_map = {
        "Fresh": "Fresh",
        "Formalin-mixed": "Pesticide"
    }

    fruitvision_count = 0
    for folder_name, mapped_label in label_map.items():
        folder_path = FRUITVISION_ROOT / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_name} not found at {folder_path}")
            continue

        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
        for i, img_path in enumerate(images):
            pair_id = f"fv_{folder_name}_{i:04d}"
            refined_pairs[pair_id] = {
                "rgb_path": str(img_path),
                "label": mapped_label
            }
            fruitvision_count += 1

    print(f"Added {fruitvision_count} samples from FruitVision (Fresh + Formalin).")

    # 3. Save the refined manifest
    manifest = {"pairs": refined_pairs}
    with open(REFINED_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully created refined manifest at {REFINED_MANIFEST_PATH}")
    print(f"Total samples in refined blind set: {len(refined_pairs)}")

if __name__ == "__main__":
    generate_refined_manifest()
