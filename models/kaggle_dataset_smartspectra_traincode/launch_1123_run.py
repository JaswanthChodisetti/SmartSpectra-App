"""
Launch cell for the 1123-scene training run.

Run this AFTER setup_1123_run.py has finished. It uses subprocess.call
with a list of args (not a shell string) so multi-line breaks and
backslashes are not a problem.

NOTE — status (2026-08-20):
  These v3 launch cells were prepared after the successful smartspectra_v2
  run (epoch=20/iter=20000, best_mrae=0.0120, see
  `models/checkpoints/run_trained/PROVENANCE.md`) but have NOT been
  executed on Kaggle. Keep them around as the recipe for a future longer
  run (100 epochs, accum_steps=4, --amp), but do NOT assume any output
  .pth exists from this v3 configuration. The currently pinned checkpoint
  is the v2 file in `models/checkpoints/run_trained/net_best.pth`.
"""
import os, subprocess, sys

WORK_TRAIN = "/kaggle/working/train_code"
os.chdir(WORK_TRAIN)

weights = os.path.join(WORK_TRAIN, "model_zoo", "mst_plus_plus.pth")
cmd = [
    sys.executable, "train.py",
    "--method", "mst_plus_plus",
    "--pretrained_model_path", weights,
    "--batch_size", "4",          # small per-step batch (Kaggle T4 = 15.6 GB VRAM)
    "--accum_steps", "4",         # effective batch 16
    "--amp",                      # mixed precision (halves activation memory)
    "--end_epoch", "100",         # ~100k iters total budget
    "--patience", "5",            # tighter early stop
    "--init_lr", "4e-4",
    "--data_root", "/kaggle/working/agro_hsr_full",
    "--outf", "/kaggle/working/exp/smartspectra_v3/",
    "--patch_size", "128",
    "--stride", "8",
    "--gpu_id", "0",
    "--lambda_smooth", "0.02",
    "--lambda_spectral", "0.02",
    "--lambda_nonneg", "0.0",      # sigmoid output — nonneg is redundant
    "--min_delta", "1e-4",
]
print("Launching:", " ".join(cmd))
ret = subprocess.call(cmd)
print(f"train.py exited with code {ret}")
