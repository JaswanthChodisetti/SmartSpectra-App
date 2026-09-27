import json
import os
from pathlib import Path

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "pesticides" / "vaishnavi_2023"
# We also search in other possible manifest locations
MANIFESTS_TO_FIX = list(ROOT.rglob("*.json"))

def fix_paths(path_str):
    if not path_str:
        return path_str

    # Target base to remove: /mnt/c/Users/jaswa/ollama/Smart_Spectra/
    # Or any variation of it
    base_patterns = [
        "/mnt/c/Users/jaswa/ollama/Smart_Spectra/",
        "/mnt/c/Users/jaswa/ollama/Smart_spectra/",
        "C:\\Users\\jaswa\\ollama\\Smart_Spectra\\",
        "C:\\Users\\jaswa\\ollama\\Smart_spectra\\",
    ]

    for pattern in base_patterns:
        if path_str.startswith(pattern):
            # Replace absolute base with relative 'data/' or similar
            # We want the path to be relative to the PROJECT ROOT
            return "data/" + path_str[len(pattern):]

    return path_str

def main():
    print(f"Scanning for manifests to fix in {ROOT}...")
    for manifest_path in MANIFESTS_TO_FIX:
        # Skip files in backup or hidden folders
        if "Backup" in str(manifest_path) or ".git" in str(manifest_path):
            continue

        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)

            changed = False
            if "pairs" in data:
                for pair_id, info in data["pairs"].items():
                    if "rgb_path" in info:
                        old = info["rgb_path"]
                        new = fix_paths(old)
                        if old != new:
                            info["rgb_path"] = new
                            changed = True
                    if "hsi_path" in info:
                        old = info["hsi_path"]
                        new = fix_paths(old)
                        if old != new:
                            info["hsi_path"] = new
                            changed = True

            if changed:
                with open(manifest_path, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Fixed paths in: {manifest_path.relative_to(ROOT)}")
            else:
                print(f"No changes needed for: {manifest_path.relative_to(ROOT)}")

        except Exception as e:
            print(f"Error fixing {manifest_path}: {e}")

if __name__ == "__main__":
    main()
