# run_trained/net_best.pth — PROVENANCE

This file is the authoritative record of what `run_trained/net_best.pth`
contains and how it was produced. Every claim in this document is either
(a) directly verifiable from the `.pth` pickle, (b) directly verifiable
from the co-located `training.log`, or (c) explicitly flagged as an
inference rather than a fact.

> Anyone editing `demo/inference.py`'s `PINNED_DEFAULT_CKPT`, anyone
> writing a report that quotes numbers from this checkpoint, and anyone
> considering re-running training MUST read this file first.

## File metadata

| Field               | Value                                                                 |
|---------------------|-----------------------------------------------------------------------|
| Path                | `models/checkpoints/run_trained/net_best.pth`                         |
| Size on disk        | 19,711,992 bytes (~18.8 MiB)                                          |
| Last modified       | 2026-08-14 01:23:43 +0530                                             |
| sha256              | `961878820979668fccd2a4c160ba67c5824d296cb7161d10b48ae6c23447912f`   |
| File format         | PyTorch pickle, top-level keys: `epoch`, `iter`, `state_dict`, `optimizer`, `best_mrae`, `cli_args` |
| Companion log       | `models/checkpoints/run_trained/training.log` (35 lines, full per-iter val curve) |
| Held-out test eval  | `experiments/test_eval/{summary.json, per_scene.json, histograms.png}` (n=282, run 2026-08-20) |

## Top-level keys (read directly from the pickle)

```
epoch      = 20
iter       = 20000
best_mrae  = tensor(0.0120)        # saved at this iter
cli_args   = {
  'method'                  : 'mst_plus_plus',
  'pretrained_model_path'   : '/kaggle/working/train_code/model_zoo/mst_plus_plus.pth',
  'batch_size'              : 10,
  'end_epoch'               : 30,
  'init_lr'                 : 0.0004,
  'outf'                    : '/kaggle/working/exp/smartspectra_v2/2026_08_13_11_59_47',
  'data_root'               : '/kaggle/working/agro_hsr_work',
  'patch_size'              : 128,
  'stride'                  : 8,
  'gpu_id'                  : '0',
  'use_l1'                  : True,
  'lambda_smooth'           : 0.02,
  'lambda_spectral'         : 0.02,
  'lambda_nonneg'           : 0.02,
  'patience'                : 10,
  'min_delta'               : 0.0001,
}
```

## State-dict

- `len(state_dict) == 227` (matches `MST_Plus_Plus`'s parameter count).
- First key: `conv_in.weight`. Last key: `conv_out.weight`.
- Same key set appears in `run_dummy/net_best.pth` (random init) —
  the architecture is identical, only the values differ.
- Sample (`conv_in.weight`): mean ≈ 0.0014, std ≈ 0.1043 — **not** random
  init (random init has std ≈ 1/sqrt(fan_in) ≈ 0.06 with smaller
  magnitude). Consistent with a fine-tune from the published MST++
  pretrained checkpoint, which `pretrained_model_path` claims.

## Run history (read directly from `training.log`)

| Item                  | Value                                       |
|-----------------------|---------------------------------------------|
| Run identifier        | `smartspectra_v2`                           |
| Outf path             | `/kaggle/working/exp/smartspectra_v2/2026_08_13_11_59_47` |
| Start (Kaggle local)  | 2026-08-13 11:59:48                         |
| First log line        | 2026-08-13 11:59:48 — patched train.py banner |
| Final log line        | 2026-08-13 19:45:46 — `EARLY STOP at iter=30000, best MRAE=0.011982` |
| Wall-clock duration   | ~7h 46m (P100/T4 GPU on Kaggle)             |
| Logged eval iters     | 1000 → 29000 in steps of 1000 (29 samples)  |
| Stop reason           | EARLY STOP at iter 30000 (patience=10 exhausted) |

### Per-iter val curve (extracted from `training.log`)

| iter   | epoch | lr        | Train Loss | Val MRAE | Val RMSE | Val PSNR (dB) |
|--------|-------|-----------|------------|----------|----------|---------------|
|  1000  |   1   | 3.989e-04 | 0.02046    | 0.02178  | 0.04489  | 27.48 |
|  2000  |   2   | 3.956e-04 | 0.01576    | 0.02064  | 0.04259  | 27.92 |
|  3000  |   3   | 3.902e-04 | 0.01362    | 0.01754  | 0.03631  | 29.20 |
|  4000  |   4   | 3.828e-04 | 0.01236    | 0.01597  | 0.03462  | 29.49 |
|  5000  |   5   | 3.733e-04 | 0.01155    | 0.01517  | 0.03343  | 29.87 |
|  6000  |   6   | 3.619e-04 | 0.01094    | 0.01544  | 0.03367  | 29.85 |
|  7000  |   7   | 3.488e-04 | 0.01047    | 0.01541  | 0.03338  | 29.96 |
|  8000  |   8   | 3.340e-04 | 0.01010    | 0.01630  | 0.03403  | 29.76 |
|  9000  |   9   | 3.178e-04 | 0.00978    | 0.01421  | 0.03168  | 30.34 |
| 10000  |  10   | 3.003e-04 | 0.00951    | 0.01414  | 0.03037  | 30.66 |
| 11000  |  11   | 2.817e-04 | 0.00927    | 0.01539  | 0.03255  | 30.11 |
| 12000  |  12   | 2.622e-04 | 0.00905    | 0.01318  | 0.02963  | 30.91 |
| 13000  |  13   | 2.420e-04 | 0.00886    | 0.01308  | 0.02944  | 30.97 |
| 14000  |  14   | 2.214e-04 | 0.00868    | 0.01269  | 0.02853  | 31.23 |
| 15000  |  15   | 2.005e-04 | 0.00851    | 0.01317  | 0.02905  | 31.07 |
| 16000  |  16   | 1.797e-04 | 0.00836    | 0.01420  | 0.03094  | 30.59 |
| 17000  |  17   | 1.590e-04 | 0.00822    | 0.01285  | 0.02958  | 30.97 |
| 18000  |  18   | 1.389e-04 | 0.00808    | 0.01223  | 0.02800  | 31.42 |
| 19000  |  19   | 1.194e-04 | 0.00795    | 0.01238  | 0.02839  | 31.33 |
| **20000** | **20** | **1.008e-04** | **0.00783** | **0.01198** ← best | **0.02756** | **31.53** |
| 21000  |  21   | 8.325e-05 | 0.00772    | 0.01280  | 0.02879  | 31.15 |
| 22000  |  22   | 6.702e-05 | 0.00761    | 0.01214  | 0.02815  | 31.37 |
| 23000  |  23   | 5.226e-05 | 0.00751    | 0.01206  | 0.02770  | 31.47 |
| 24000  |  24   | 3.911e-05 | 0.00742    | 0.01232  | 0.02823  | 31.35 |
| 25000  |  25   | 2.774e-05 | 0.00733    | 0.01198  | 0.02792  | 31.44 |
| 26000  |  26   | 1.826e-05 | 0.00724    | 0.01208  | 0.02796  | 31.43 |
| 27000  |  27   | 1.077e-05 | 0.00716    | 0.01199  | 0.02789  | 31.46 |
| 28000  |  28   | 5.364e-06 | 0.00708    | 0.01206  | 0.02799  | 31.42 |
| 29000  |  29   | 2.095e-06 | 0.00700    | 0.01201  | 0.02788  | 31.46 |
| STOP   |       |           |            | 0.01198 (best) |        |        |

### Why the saved `.pth` is `epoch=20 / iter=20000`, not `iter=30000`

The trainer saves a checkpoint every time val-MRAE improves. The val-MRAE
curve shows the **lowest value at iter=20000** (0.01198, exactly matching
`best_mrae=0.0120` in the pickle). Iters 21000–29000 fluctuate between
0.01198 and 0.01280, never strictly beating 0.01198, so no new "best"
checkpoint was written. The `EARLY STOP` line confirms the trainer's
view: `best MRAE=0.011982` (the iter=20000 value) was the best seen.

**The `.pth` is correctly the best checkpoint of the run.** Its `epoch=20
/ iter=20000` fields simply reflect when the save fired, not the end of
training.

## What this file does **not** prove

1. **The dataset split used in `agro_hsr_work/` matches the local
   `data/raw/Agro-HSR/split_txt/`.** `data_root` in the pickle is the
   Kaggle path. Verify by checking `len(train_list) == 1123` and that
   the stems match what the run actually trained on (not preserved in
   the pickle).
2. **The `training.log` was captured without buffering/loss.** It is the
   only copy we have; if it is incomplete or buffered-truncated, the
   curve above is wrong. The fact that we have 29 evenly-spaced eval
   rows from iter=1000 to iter=29000 (no missing iters) suggests a
   complete capture, but we cannot independently verify.

## How to inspect / re-verify

```python
import torch, hashlib
p = "models/checkpoints/run_trained/net_best.pth"
print(hashlib.sha256(open(p, "rb").read()).hexdigest())
obj = torch.load(p, map_location="cpu", weights_only=False)
print({k: obj[k] for k in obj if k != "state_dict"})
print("len(state_dict):", len(obj["state_dict"]))
```

### Held-out test-set evaluation (read from `experiments/test_eval/summary.json`)

282 Agro-HSR test scenes (the model has never seen these). Run via
`scripts/eval_test_set.py`. Numbers below are the full-testset result, not
the n=6 spot-check in `demo/hsi_samples/batch_eval.json`.

| Metric | Mean    | Median  | Std     | Min     | Max     |
|--------|---------|---------|---------|---------|---------|
| MRAE   | 0.0261  | 0.0236  | 0.0106  | 0.0100  | 0.0749  |
| RMSE   | 0.0115  | 0.0113  | 0.0025  | 0.0060  | 0.0216  |
| PSNR   | 37.21 dB| 37.29 dB| 2.21 dB | 30.51 dB| 42.44 dB|

Top-5 worst scenes by MRAE: Test_64, Test_167, Test_175, Test_30, Test_128.
Top-5 best scenes by MRAE:  Test_215, Test_242, Test_191, Test_196, Test_200.

**Reading the gap (0.0120 → 0.0261):** The training-val MRAE of 0.0120
is on scenes the model has gradient-updated against. The 282-scene test
MRAE of 0.0261 is on held-out scenes — that's the gap between "fits the
training distribution" and "generalises". This is the number to cite
in any report or presentation.

## How to regenerate

To regenerate a checkpoint with known provenance:

1. Use the Phase-2 cell bundle at
   `notebooks/phase2_kaggle_cells.py` (it ships a patched
   `hsi_dataset.py`, `utils.py`, `train.py`, and a launch cell).
2. On Kaggle, capture the **entire stdout** of `train.py` to a log
   file via `tee` (e.g. `python train.py ... 2>&1 | tee training.log`)
   and save `training.log` **next to** `net_best.pth` (i.e. in this
   directory, not in `kaggle_dataset_smartspectra_traincode/`).
3. Update this PROVENANCE.md with the new run's timestamp, epoch,
   best_mrae, and a pointer to the new training log.

This v2 run followed that recipe; the log is now co-located with the
`.pth` and the numbers above are pulled from it.
