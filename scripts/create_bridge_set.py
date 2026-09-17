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
REFINED_MANIFEST_PATH = ROOT / "data/pesticides/apple_refined_blind_manifest.json"
NEW_REFINED_MANIFEST = ROOT / "data/pesticides/apple_final_blind_manifest.json"

# Configuration
SAMPLES_PER_CLASS = 100 # Number of images to move to training from each class

def create_bridge_set():
    print("--- STARTING DOMAIN ADAPTATION: BRIDGE SET CREATION ---")

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

    moved_images = set()

    for label, src_folder in source_map.items():
        if not src_folder.exists():
            print(f"Warning: Source folder {src_folder} not found. Skipping {label}.")
            continue

        # Get all images and shuffle them
        images = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpeg"))
        random.shuffle(images)

        # Take the first N samples
        selection = images[:SAMPLES_PER_CLASS]
        target_folder = target_map[label]
        target_folder.mkdir(parents=True, exist_ok=True)

        print(f"Moving {len(selection)} {label} images to clean_dataset...")
        for img_path in selection:
            # Use original filename to avoid collisions
            dest_path = target_folder / img_path.name
            shutil.copy(str(img_path), str(dest_path))
            # Store the absolute path for manifest removal
            moved_images.add(str(img_path.absolute()))

    print(f"Successfully moved {len(moved_images)} images to the training set.")

    # 2. Update the Manifest to remove these images (Prevent Data Leakage)
    print("Updating blind manifest to prevent data leakage...")

    # Load the current refined manifest
    if REFINED_MANIFEST_PATH.exists():
        with open(REFINED_MANIFEST_PATH, "r") as f:
            manifest_data = json.load(f)

        pairs = manifest_data.get("pairs", {})
        initial_count = len(pairs)

        # Filter out any image that was moved to training
        new_pairs = {
            pid: info for pid, info in pairs.items()
            if Path(info["rgb_path"]).absolute() not in moved_images
        }

        removed_count = initial_count - len(new_pairs)
        manifest_data["pairs"] = new_pairs

        with open(NEW_REFINED_MANIFEST, "w") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"Removed {removed_count} samples from blind manifest to prevent leakage.")
        print(f"Final blind set size: {len(new_pairs)}")
        print(f"New manifest saved to {NEW_REFINED_MANIFEST}")
    else:
        print("Error: Refined manifest not found. Could not update test set.")

if __name__ == "__main__":
    create_bridge_set()
