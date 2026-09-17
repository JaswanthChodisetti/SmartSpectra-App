"""
One-cell verifier for the Agro-HSR train/valid split on Kaggle.

Run this in a notebook cell before launching the 1123-scene training
run. Catches the two ways the big run can silently fail:

  1. The split was generated against a partial download (missing files).
  2. The split paths are inconsistent between the local copy and the
     Kaggle copy.

Usage (Kaggle notebook cell):
    !python /kaggle/working/verify_split.py --data_root /kaggle/input/agro-hsr/

The script prints PASS/FAIL with concrete numbers. If it returns non-zero,
do not launch the training run.
"""
import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Path to the Agro-HSR root (the dir that has "
                         "Train_Spec/, Test_Spec/, Train_RGB/, Test_RGB/, "
                         "split_txt/).")
    args = ap.parse_args()
    root = Path(args.data_root)

    hyper_dirs = [root / "Train_Spec", root / "Test_Spec"]
    rgb_dirs = [root / "Train_RGB", root / "Test_RGB"]
    for d in hyper_dirs + rgb_dirs:
        if not d.is_dir():
            print(f"FAIL: missing dir {d}")
            return 2

    train_list = root / "split_txt" / "train_list.txt"
    valid_list = root / "split_txt" / "valid_list.txt"
    if not train_list.is_file() or not valid_list.is_file():
        print(f"FAIL: missing {train_list} or {valid_list}")
        return 2

    train_stems = train_list.read_text().splitlines()
    valid_stems = valid_list.read_text().splitlines()
    n_train, n_valid = len(train_stems), len(valid_stems)
    total = n_train + n_valid
    print(f"train_list.txt: {n_train} entries")
    print(f"valid_list.txt: {n_valid} entries")
    print(f"total: {total} (expected 1322)")

    if total != 1322:
        print(f"FAIL: expected 1322 total scenes, got {total}")
        return 3
    if n_train < 1100:
        print(f"FAIL: train set has {n_train} scenes; expected ~1123. "
              f"Did someone re-run prep_agrohsr_split.py with a smaller "
              f"source download?")
        return 3
    if n_valid < 150:
        print(f"FAIL: valid set has {n_valid} scenes; expected ~199.")
        return 3

    overlap = set(train_stems) & set(valid_stems)
    if overlap:
        print(f"FAIL: {len(overlap)} stems appear in BOTH train and valid. "
              f"First 3: {sorted(overlap)[:3]}")
        return 4

    # Every stem must resolve to a .mat in Train_Spec/ or Test_Spec/ and
    # a .jpg in Train_RGB/ or Test_RGB/.
    missing_mat: list[str] = []
    missing_jpg: list[str] = []
    for stem in train_stems + valid_stems:
        mat = next((d / f"{stem}.mat" for d in hyper_dirs
                    if (d / f"{stem}.mat").is_file()), None)
        jpg = next((d / f"{stem}.jpg" for d in rgb_dirs
                    if (d / f"{stem}.jpg").is_file()), None)
        if mat is None:
            missing_mat.append(stem)
        if jpg is None:
            missing_jpg.append(stem)

    if missing_mat:
        print(f"FAIL: {len(missing_mat)} stems have no .mat on disk. "
              f"First 3: {missing_mat[:3]}")
        return 5
    if missing_jpg:
        print(f"FAIL: {len(missing_jpg)} stems have no .jpg on disk. "
              f"First 3: {missing_jpg[:3]}")
        return 5

    # Estimate one-cube resident memory (warn if the cubes are unusually
    # big, which would blow the streaming budget on Kaggle).
    sample = next(d / f"{train_stems[0]}.mat" for d in hyper_dirs
                  if (d / f"{train_stems[0]}.mat").is_file())
    sample_bytes = sample.stat().st_size
    sample_mb = sample_bytes / (1024 * 1024)
    print(f"Sample .mat size ({train_stems[0]}): {sample_mb:.1f} MB")
    if sample_mb > 200:
        print(f"WARN: cubes are larger than expected ({sample_mb:.1f} MB). "
              f"Streaming still works but consider a smaller patch_size.")

    print("PASS: split is well-formed and all stems resolve on disk.")
    print("Safe to launch the 1123-scene training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
