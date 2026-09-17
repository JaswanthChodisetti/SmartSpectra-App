# SmartSpectra — SpectroFood apple pairs + finetune script

This dataset packages the 240 (rgb, hsi_31band) pairs synthesized from
the SpectroFood Apple.mat cube library (Malounas et al., 2023), along
with the MST++ architecture and the CLI-refactored fine-tune script.

Use it in any Kaggle notebook via "+ Add Input" — search for
`smartspectra-spectrofood`.

## Layout

    smartspectra-spectrofood/
      architecture/MST_Plus_Plus.py
      scripts/train_model1_apple_finetune.py
      scripts/cie1931_srgb.py
      scripts/build_vaishnavi_manifest.py
      scripts/agro_hsr_bands.py
      apple_pairs/manifest.json
      apple_pairs/rgb/A1.png ... A240.png   (1.0 MB total)
      apple_pairs/hsi/A1.npy ... A240.npy   (103.6 MB total)

## Pair stats

- 240 pairs of (synthesized RGB, 31-band HSI) in [0, 1] reflectance
- HSI bands: 400-1000 nm at 20 nm spacing (Agro-HSR convention)
- Binned from 141 source bands (430-990 nm @ 4 nm) per cube
- Per-pixel min-max normalised (SpectroFood has no white reference)

## Training command

    cd /kaggle/input/smartspectra-spectrofood/scripts
    python train_model1_apple_finetune.py \
        --data_root /kaggle/input/smartspectra-spectrofood/apple_pairs \
        --ckpt_in   /kaggle/input/<your-source-ckpt-dataset>/net_best.pth \
        --out_dir   /kaggle/working \
        --epochs    3   --lr 3e-5  --batch 4 \
        --data_label 'SpectroFood apple (240 cubes)'

Expected runtime: ~10-15 min on T4x2 for 3 epochs.
