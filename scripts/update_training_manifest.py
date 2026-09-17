import os
import pandas as pd
from pathlib import Path

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATASET_ROOT = ROOT / "data/clean_dataset"
MANIFEST_PATH = ROOT / "Archive/experiments/clean_apple_pesticide_manifest.csv"

def regenerate_manifest():
    print("--- REGENERATING TRAINING MANIFEST ---")

    data = []

    # Scan folders
    for label_folder, label_name in [("fresh", "Fresh"), ("pesticide", "Pesticide")]:
        folder_path = CLEAN_DATASET_ROOT / label_folder
        if not folder_path.exists():
            print(f"Warning: Folder {label_folder} not found.")
            continue

        images = []
        # Support common image extensions
        for ext in ["*.tif", "*.TIF", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
            images.extend(list(folder_path.glob(ext)))

        print(f"Found {len(images)} images in {label_folder} folder.")

        for img_path in images:
            data.append({
                "path": str(img_path),
                "label": label_name
            })

    if not data:
        print("Error: No images found in clean_dataset folders.")
        return

    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nSuccessfully updated manifest at {MANIFEST_PATH}")
    print(f"Total images in training manifest: {len(df)}")

if __name__ == "__main__":
    regenerate_manifest()
