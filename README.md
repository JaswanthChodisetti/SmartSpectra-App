# SmartSpectra

Reconstruct hyperspectral (HSI) image data from smartphone RGB photos of fruit, then
use that reconstructed HSI to screen for pesticide residue risk.

**Framing**: Low-cost pre-screening / triage tool — NOT a replacement for certified
hyperspectral cameras or lab-based pesticide testing. The NIR (~915-1699 nm) gap
between smartphone RGB and diagnostic pesticide bands is a hardware physics
constraint, not a fixable limitation. We measure the accuracy gap to real HSI and
to published RGB-only baselines; we do not claim to "solve" it.

---

## Environment split

- **Laptop (RTX 3050 Ti, 4 GB VRAM)**: data prep, inference, deployment app.
  No heavy model training here.
- **Kaggle (free P100 / 2×T4, 30 hrs/wk)**: MST++ fine-tuning, joint training.

You write code locally; I (Claude) output ready-to-paste Kaggle cells for any
training work, and you run them there.

---

## Project structure

```
Smart_Spectra/
├── README.md                   # this file
├── PLAN.md                     # phase-by-phase plan
├── TASKS.md                    # running checklist, updated as phases complete
├── data/
│   ├── raw/                    # dataset roots (see Data layout below)
│   │   ├── Agro-HSR/           # symlink → /mnt/c/.../Downloads/Agro-HSR
│   │   │   ├── Train_Spec/*.mat
│   │   │   ├── Train_RGB/*.jpg
│   │   │   ├── Test_Spec/*.mat
│   │   │   ├── Test_RGB/*.jpg
│   │   │   └── split_txt/{train,valid}_list.txt
│   │   ├── Train_Spec/*.mat     # direct copies (what prep_agrohsr_split.py writes)
│   │   ├── Train_RGB/*.jpg
│   │   ├── Test_Spec/*.mat
│   │   ├── Test_RGB/*.jpg
│   │   └── split_txt/{train,valid}_list.txt
│   ├── processed/
│   └── pesticides/             # banana/apple/okra/grape pesticide datasets (later)
├── code/
│   └── MST-plus-plus/          # MST++ repo (cloned, used for training + inference)
├── scripts/
│   ├── prep_agrohsr_split.py       # Phase 1: hard-copies + 85/15 split
│   ├── verify_agrohsr_split.py     # Phase 1: sanity-check the converted layout
│   ├── inspect_agrohsr.py          # Phase 1: probe one .mat's structure
│   └── inspect_dataset.py          # Phase 1: generic format detector
├── notebooks/                  # local Jupyter scratch
├── experiments/                # experiment tracking (results, plots, metrics)
├── models/                     # local checkpoints (Phase 2 lights back from Kaggle)
└── reports/                    # Capstone proposal, final report, slides
```

### Data layout (Agro-HSR)

The dataset lives in **two parallel layouts** under `data/raw/`:

| Layout | Path | Produced by | Used by |
|--------|------|-------------|---------|
| Direct (legacy / current) | `data/raw/{Train,Test}_{RGB,Spec}/` | `prep_agrohsr_split.py` (hard-copies) | demo, all scripts |
| Symlink (canonical per this README) | `data/raw/Agro-HSR/{Train,Test}_{RGB,Spec}/` | manual symlink to the raw download | prep_agrohsr_split.py source candidates |

`prep_agrohsr_split.py` hard-copies (not symlinks) so re-running it
gives you the legacy layout regardless of the symlink target. The demo
and inference helpers (`demo/inference.py::TEST_RGB_DIR`) look at the
legacy layout first, then fall back to the canonical one. Both work.

---

## Pipeline

```
smartphone RGB photo
   -> background segmentation (rembg / GrabCut)              [Phase 5]
   -> Model 1: MST++ (fine-tuned on Agro-HSR) reconstruct HSI  [Phase 2-3]
   -> preprocessing (SNV, Savitzky-Golay)                   [Phase 4]
   -> band-reliability feature selection (SPA/CARS)          [Phase 4]
   -> Model 2: SVM/RandomForest on RECONSTRUCTED HSI         [Phase 4]
   -> calibrated confidence (CalibratedClassifierCV)         [Phase 4]
   -> low confidence? -> recommend lab testing              [Phase 5]
```

---

## Key permanent constraints

- **Smartphone RGB cannot capture NIR (~915-1699 nm)**. This is a hardware
  physics constraint (IR-cut filter + silicon sensor limit). It is not a
  software/training problem. Our pesticide datasets span 400–1000 nm, so we
  keep the visible + red-edge NIR edge, but lose the most chemically diagnostic
  bands. **Never claim this has been "solved" in code, comments, or reports.**
- **RGB → HSI is fundamentally ill-posed** (metamerism: many real spectra map
  to the same RGB). We measure the reconstruction error honestly, not as a
  solved problem.
- **Accuracy numbers must come from actually running the pipeline.** No
  fabricated or assumed values in code or docs.

---

## References

- Cai et al., *MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral
  Reconstruction*, CVPRW 2022 — github.com/caiyuanhao1998/MST-plus-plus
- Shi et al., *HSCNN+: Advanced CNN-Based Hyperspectral Recovery from RGB Images*,
  CVPRW 2018
- Agro-HSR dataset (real produce RGB↔HSI pairs, 31 bands, 400-1000 nm)
