import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_filename(filename):
    """
    Maps filename to pesticide type and concentration based on:
    - 'ma' -> Insecticide
    - 'a'  -> Fungicide (but not 'ma')
    - 'l'  -> Low Concentration
    - 'h'  -> High Concentration
    """
    fname = filename.lower()

    # Type mapping
    p_type = "Fresh" # Default
    if "ma" in fname:
        p_type = "Insecticide"
    elif "a" in fname:
        p_type = "Fungicide"

    # Concentration mapping
    conc = ""
    if "l" in fname:
        conc = "Low Concentration"
    elif "h" in fname:
        conc = "High Concentration"

    # Final combined label
    if p_type == "Fresh":
        return "Fresh", "Fresh"

    detailed_label = f"{conc} {p_type}".strip()
    return "Pesticide", detailed_label

def create_manifest(data_dir, output_csv):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    files = [f for f in data_path.iterdir() if f.is_file()]
    print(f"Found {len(files)} files in {data_dir}")

    data = []
    for f in tqdm(files, desc="Mapping files"):
        binary_label, detailed_label = parse_filename(f.name)
        data.append({
            "path": str(f.absolute()),
            "label": binary_label,
            "detailed_label": detailed_label,
            "filename": f.name
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Manifest saved to {output_csv}")

    # Summary
    print("\nDistribution Summary:")
    print(df["detailed_label"].value_counts())

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python map_new_dataset.py <data_dir> <output_csv>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    create_manifest(input_dir, output_file)
