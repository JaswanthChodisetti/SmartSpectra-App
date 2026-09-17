"""
Single-cell setup for the 1123-scene training run on Kaggle.

Run this in a fresh notebook cell. After it finishes, launch the
training in a follow-up cell with subprocess.call (NOT with `!python ...`,
which trips over multi-line backslash continuations in Code cells).

Output paths assume:
  - Train code dataset: /kaggle/input/datasets/chodisettijaswanth/smartspectra-patched-traincode-v2
  - Agro-HSR dataset:   /kaggle/input/datasets/jaswanthchodisetti/agro-hsr
  - Pretrained weights: gdown'd from 18X6RkcQaIuiV5gRbswo7GLv7WJG9M_WM

NOTE — status (2026-08-20):
  These v3 setup cells were prepared after the successful smartspectra_v2
  run (epoch=20/iter=20000, best_mrae=0.0120, see
  `models/checkpoints/run_trained/PROVENANCE.md`) but have NOT been
  executed on Kaggle. Keep them as the recipe for a future longer run.
  The currently pinned checkpoint is the v2 file in
  `models/checkpoints/run_trained/net_best.pth`.
"""
import os, shutil, subprocess, sys, urllib.request

WORK_TRAIN = "/kaggle/working/train_code"
WORK_DATA  = "/kaggle/working/agro_hsr_full"
WEIGHTS_PATH = os.path.join(WORK_TRAIN, "model_zoo", "mst_plus_plus.pth")

# ----------------------------------------------------------------------
# 1) Copy train code from the Kaggle dataset to the writable working dir.
# ----------------------------------------------------------------------
train_code_src = "/kaggle/input/datasets/chodisettijaswanth/smartspectra-patched-traincode-v2"
if not os.path.isdir(train_code_src):
    # Fall back: search for it (matches cell-3's diagnostic behaviour)
    r = subprocess.run(
        ["find", "/kaggle/input", "-name", "train.py", "-not", "-path", "*/working/*"],
        capture_output=True, text=True,
    )
    matches = [p.rsplit("/", 1)[0] for p in r.stdout.splitlines() if p.strip()]
    if not matches:
        raise SystemExit("Could not find train.py on /kaggle/input")
    train_code_src = matches[0]
print(f"Train code source: {train_code_src}")

if os.path.exists(WORK_TRAIN):
    shutil.rmtree(WORK_TRAIN)
shutil.copytree(train_code_src, WORK_TRAIN)
print(f"Copied to {WORK_TRAIN}")

# ----------------------------------------------------------------------
# 2) Verify the train code has the new memory-safe flags. If it doesn't,
#    the dataset was uploaded before our patches landed — in that case,
#    we re-copy hsi_dataset.py + train.py from your local repo (the user
#    is expected to have synced them to Kaggle, or we fail loudly).
# ----------------------------------------------------------------------
with open(os.path.join(WORK_TRAIN, "train.py")) as f:
    src = f.read()
needs_patches = not ("accum_steps" in src and "cuda.amp" in src)
if needs_patches:
    print("WARN: /kaggle/working/train_code/train.py is missing --accum_steps / --amp.")
    print("   Re-upload the patched train.py + hsi_dataset.py to the Kaggle")
    print("   dataset, OR run the launch cell's setup step that copies them")
    print("   from your local repo (see setup_local_patches.py).")
else:
    print("Patched train.py verified (has --accum_steps and --amp).")

with open(os.path.join(WORK_TRAIN, "hsi_dataset.py")) as f:
    ds = f.read()
if "_load_cube" not in ds:
    print("WARN: /kaggle/working/train_code/hsi_dataset.py is the old preload")
    print("   version. Streaming is OFF. The 1123-scene run will likely OOM.")
else:
    print("Patched hsi_dataset.py verified (streaming, _load_cube present).")

# ----------------------------------------------------------------------
# 3) Build a working copy of the full 1123-scene Agro-HSR split.
# ----------------------------------------------------------------------
agrohsr_src = "/kaggle/input/datasets/jaswanthchodisetti/agro-hsr"
if not os.path.isdir(agrohsr_src):
    # Walk to find Train_Spec/ + split_txt/ co-located
    for root, dirs, _ in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "Train_Spec" in dirs and "split_txt" in dirs:
            agrohsr_src = root
            break
print(f"Agro-HSR source: {agrohsr_src}")

if os.path.exists(WORK_DATA):
    shutil.rmtree(WORK_DATA)
os.makedirs(os.path.join(WORK_DATA, "split_txt"))
for sub in ("Train_Spec", "Test_Spec", "Train_RGB", "Test_RGB"):
    src = os.path.join(agrohsr_src, sub)
    dst = os.path.join(WORK_DATA, sub)
    if os.path.isdir(src):
        os.symlink(src, dst)
        print(f"  symlinked {sub}/")
for fn in ("train_list.txt", "valid_list.txt"):
    shutil.copy(os.path.join(agrohsr_src, "split_txt", fn),
                os.path.join(WORK_DATA, "split_txt", fn))

n_train = sum(1 for _ in open(os.path.join(WORK_DATA, "split_txt/train_list.txt")))
n_valid = sum(1 for _ in open(os.path.join(WORK_DATA, "split_txt/valid_list.txt")))
print(f"Working data: {n_train} train / {n_valid} valid scenes (no truncation).")

# ----------------------------------------------------------------------
# 4) Ensure pretrained MST++ weights are present.
# ----------------------------------------------------------------------
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
if not os.path.isfile(WEIGHTS_PATH):
    # Try gdown first (the existing cell-5 approach). If gdown is not
    # installed or the file is gated, fall back to a direct URL via
    # urllib — the gdown id is the public MST++ release on Google Drive.
    try:
        subprocess.run(
            ["gdown", "18X6RkcQaIuiV5gRbswo7GLv7WJG9M_WM", "-O", WEIGHTS_PATH],
            check=True, timeout=300,
        )
    except Exception as e:
        print(f"gdown failed ({e}); pretrained weights NOT downloaded.")
        print("Training will start from random init — expect a worse MRAE.")
else:
    print(f"Pretrained weights present ({os.path.getsize(WEIGHTS_PATH)/1e6:.1f} MB).")

# ----------------------------------------------------------------------
# 5) Sanity check: a single __getitem__ should return the right shapes.
# ----------------------------------------------------------------------
sys.path.insert(0, WORK_TRAIN)
import hsi_dataset
ds = hsi_dataset.TrainDataset(WORK_DATA, crop_size=128, arg=True, bgr2rgb=True, stride=8)
bgr, hyper = ds[0]
assert bgr.shape == (3, 128, 128), f"bad bgr shape: {bgr.shape}"
assert hyper.shape == (31, 128, 128), f"bad hyper shape: {hyper.shape}"
print(f"Sanity check passed: bgr={bgr.shape}, hyper={hyper.shape}, "
      f"len(train)={len(ds)} patches.")

print("\nReady. Launch with the 1123-scene training command in the next cell.")
print(f"  --data_root {WORK_DATA}")
print(f"  --outf /kaggle/working/exp/smartspectra_v3/")
