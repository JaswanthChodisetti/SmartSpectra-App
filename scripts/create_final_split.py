import os
import random
import shutil
import json
from pathlib import Path

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
FRUITVISION_ROOT = ROOT / "data/Fruit_Vision/extracted/Fruits Original/Apple"
CLEAN_DATASET_ROOT = ROOT / "data/clean_dataset"
VAISHNAVI_BLIND = ROOT / "data/pesticides/vaishnavi_2023/apple_pairs_true_blind.json"
FINAL_BLIND_MANIFEST = ROOT / "data/pesticides/apple_final_blind_manifest.json"

# Configuration
TEST_SAMPLES_PER_CLASS = 500 # The a-priori decided hold-out set

def create_final_split():
    print("--- STARTING FINAL DOMAIN ADAPTATION SPLIT ---")

    # 1. Define the classes and their source folders
    source_map = {
        "Fresh": FRUITVISION_ROOT / "Fresh",
        "Pesticide": FRUITVISION_ROOT / "Formalin-mixed"
    }

    # Define target folders in clean_dataset
    target_map = {
        "Fresh": CLEAN_DATASET_ROOT / "fresh",
        "Pesticide": CLEAN_DATASET_ROOT / "pesticide"
    }

    blind_pairs = {}

    # Start with the original Vaishnavi blind set (the absolute gold standard)
    if VAISHNAVI_BLIND.exists():
        with open(VAISHNAVI_BLIND, "r") as f:
            vaishnavi_data = json.load(f)
            blind_pairs.update(vaishnavi_data.get("pairs", {}))
        print(f"Added {len(vaishnavi_data.get('pairs', {}))} samples from Vaishnavi blind set.")

    for label, src_folder in source_map.items():
        if not src_folder.exists():
            print(f"Warning: Source folder {src_folder} not found. Skipping {label}.")
            continue

        # Get all images
        images = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpeg"))
        random.shuffle(images)

        # Split: first N for blind test, the REST for training
        test_selection = images[:TEST_SAMPLES_PER_CLASS]
        train_selection = images[TEST_SAMPLES_PER_CLASS:]

        # Move training samples to clean_dataset
        target_folder = target_map[label]
        target_folder.mkdir(parents=True, exist_ok=True)

        print(f"Moving {len(train_selection)} {label} images to training set...")
        for img_path in train_selection:
            dest_path = target_folder / img_path.name
            shutil.copy(str(img_path), str(dest_path))

        # Add test samples to the blind manifest
        for i, img_path in enumerate(test_selection):
            pair_id = f"fv_{label}_{i:04d}"
            blind_pairs[pair_id] = {
                "rgb_path": str(img_path),
                "label": "Fresh" if label == "Fresh" else "Pesticide"
            }
        print(f"Added {len(test_selection)} {label} samples to the blind manifest.")

    # Save the final manifest
    manifest = {"pairs": blind_pairs}
    with open(FINAL_BLIND_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSUCCESS!")
    print(f"Training Set: All remaining FruitVision images moved to clean_dataset.")
    print(f"Blind Test Set: {len(blind_pairs)} samples saved to {FINAL_BLIND_MANIFEST}")

if __name__ == "__main__":
    create_final_split()
