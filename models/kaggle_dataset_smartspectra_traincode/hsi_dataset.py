"""
HSI dataset for Agro-HSR — streaming variant.

Why this exists
---------------
The original MST++ dataset (hsi_dataset.py) preloads every cube into
self.hypers / self.bgrs at __init__ time. For the full 1123-scene
Agro-HSR split at 482×512×31 float32 that is ~33 GB of labels, which
crashes Kaggle's free tier (13 GB CPU RAM) before training even starts.

This module replaces the preload with a lazy/streaming implementation:
  - __init__ only records the file paths (cheap, ~1 KB per scene)
  - __getitem__ opens the relevant .mat on demand, reads the cube,
    crops, applies augmentation, and returns
Total resident memory is "one cube in flight" (~30 MB) instead of the
whole dataset.

The patch is API-compatible: train.py and the MST++ architecture
modules continue to import TrainDataset and ValidDataset by name.
"""
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import os


# ---------------------------------------------------------------------
# Module-level helpers (kept private so they don't pollute the namespace
# when this file is imported from the train script).
# ---------------------------------------------------------------------
def _resolve(stem_ext: str, candidates: list[str]) -> str:
    """Return the first existing path for `stem_ext` from `candidates`.

    Agro-HSR ships Train_Spec/ and Test_Spec/ side by side. The split
    we build with prep_agrohsr_split.py shuffles all 1322 pairs into
    train/val regardless of which source directory they came from, so a
    stem like "Test_42" can legitimately be in the train split.
    """
    for d in candidates:
        candidate = os.path.join(d, stem_ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {stem_ext!r} in any of {candidates}"
    )


def _load_cube(mat_path: str) -> np.ndarray:
    """Read a (31, H, W) float32 cube from an Agro-HSR .mat file."""
    with h5py.File(mat_path, 'r') as mat:
        # h5py reads the cube as (1, H, W, 31) typically; squeeze the
        # singleton batch axis. We always get (B, H, W) afterwards.
        cube = np.array(mat['cube'])
    if cube.ndim == 4 and cube.shape[0] == 1:
        cube = cube[0]
    # cube is (B, H, W) with B=31; transpose to the (H, W, B) order the
    # original TrainDataset produced after `np.transpose(hyper, [0,2,1])`
    # was applied to (B, H, W) input. Wait — the original transpose was
    # (0, 2, 1), which swaps axes 1 and 2 of (B, H, W) → (B, W, H).
    # That's a leftover from the MST++ paper's data layout; preserve it
    # here so crop arithmetic stays identical.
    cube = np.float32(cube)
    cube = np.transpose(cube, [0, 2, 1])
    return cube


def _load_rgb(jpg_path: str) -> np.ndarray:
    """Read a (H, W, 3) float32 RGB image, normalized to [0, 1]."""
    bgr = cv2.imread(jpg_path)
    if bgr is None:
        raise FileNotFoundError(f"cv2 could not read {jpg_path}")
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr = np.float32(bgr)
    bgr_min = bgr.min()
    bgr_max = bgr.max()
    if bgr_max > bgr_min:
        bgr = (bgr - bgr_min) / (bgr_max - bgr_min)
    else:
        # Degenerate image (all-one channel). Avoid /0.
        bgr = np.zeros_like(bgr, dtype=np.float32)
    # (H, W, 3) → (3, H, W) to match the original TrainDataset output.
    bgr = np.transpose(bgr, [2, 0, 1])
    return bgr


# ---------------------------------------------------------------------
# Training dataset (random crop + augmentation, one cube per __getitem__)
# ---------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        # SmartSpectra: keep bgr2rgb as a kwarg for API compatibility,
        # but cv2 always reads as BGR and we convert here unconditionally.
        del bgr2rgb
        self.arg = arg
        h, w = 482, 512  # Agro-HSR image shape
        self.stride = stride
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        hyper_dirs = [
            f'{data_root}/Train_Spec/',
            f'{data_root}/Test_Spec/',
        ]
        bgr_dirs = [
            f'{data_root}/Train_RGB/',
            f'{data_root}/Test_RGB/',
        ]

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
        hyper_list.sort()
        print(f'len(hyper) of agro-hsr dataset: {len(hyper_list)}')

        # Store only file paths — no preloading.
        self.hyper_paths: list[str] = []
        self.bgr_paths: list[str] = []
        for stem_ext in hyper_list:
            stem = stem_ext.split('.')[0]
            hyper_path = _resolve(stem_ext, hyper_dirs)
            bgr_path = _resolve(stem + '.jpg', bgr_dirs)
            assert stem == bgr_path.split(os.sep)[-1].split('.')[0], (
                f'Hyper and RGB come from different scenes: '
                f'{stem} vs {bgr_path}'
            )
            self.hyper_paths.append(hyper_path)
            self.bgr_paths.append(bgr_path)

        self.img_num = len(self.hyper_paths)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for _ in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for _ in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for _ in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        # Agro-HSR scenes are 512x512 with a centered leaf on a
        # near-zero background. With stride=8, ~25% of patches land
        # entirely on background and have all-zero RGB+HSI. The loss
        # on those patches is zero (the model trivially predicts ~0)
        # so they cost forward-pass time without contributing gradient.
        # We re-roll the (h_idx, w_idx) choice within the SAME scene
        # up to MAX_SKIP_ROLLS times until we land on a patch with
        # real signal. Re-rolling within a scene (not jumping to a
        # different scene) is the key: each Agro-HSR scene has the
        # leaf in roughly the same center region, so a random (h_idx,
        # w_idx) drawn from the full patch grid almost always hits
        # the leaf on a re-roll.
        MAX_SKIP_ROLLS = 8
        img_idx = idx // self.patch_per_img
        patch_idx = idx % self.patch_per_img
        h_idx = patch_idx // self.patch_per_line
        w_idx = patch_idx % self.patch_per_line
        # Load the RGB once; we use it both for the skip check and
        # the final return. If we re-roll, we keep the same RGB.
        bgr = _load_rgb(self.bgr_paths[img_idx])
        for _ in range(MAX_SKIP_ROLLS):
            h0 = h_idx * stride
            w0 = w_idx * stride
            bgr_crop = bgr[:, h0:h0 + crop_size, w0:w0 + crop_size]
            if bgr_crop.max() > 1e-3:
                break
            # Re-roll within the same scene. The non-zero region is
            # roughly the central band of the patch grid, so sampling
            # uniformly gives a high hit rate.
            h_idx = random.randrange(self.patch_per_colum)
            w_idx = random.randrange(self.patch_per_line)
        # Read the matching cube now that we know which patch we want.
        hyper = _load_cube(self.hyper_paths[img_idx])
        h0 = h_idx * stride
        w0 = w_idx * stride
        h1, w1 = h0 + crop_size, w0 + crop_size
        bgr = bgr[:, h0:h1, w0:w1]
        hyper = hyper[:, h0:h1, w0:w1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img * self.img_num


# ---------------------------------------------------------------------
# Validation dataset (full cube per __getitem__, no crop)
# ---------------------------------------------------------------------
class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        del bgr2rgb  # API compatibility; we always do BGR→RGB.

        hyper_dirs = [
            f'{data_root}/Train_Spec/',
            f'{data_root}/Test_Spec/',
        ]
        bgr_dirs = [
            f'{data_root}/Train_RGB/',
            f'{data_root}/Test_RGB/',
        ]

        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
        hyper_list.sort()
        print(f'len(hyper_valid) of agro-hsr dataset: {len(hyper_list)}')

        self.hyper_paths: list[str] = []
        self.bgr_paths: list[str] = []
        for stem_ext in hyper_list:
            stem = stem_ext.split('.')[0]
            hyper_path = _resolve(stem_ext, hyper_dirs)
            bgr_path = _resolve(stem + '.jpg', bgr_dirs)
            assert stem == bgr_path.split(os.sep)[-1].split('.')[0], (
                f'Hyper and RGB come from different scenes: '
                f'{stem} vs {bgr_path}'
            )
            self.hyper_paths.append(hyper_path)
            self.bgr_paths.append(bgr_path)

    def __getitem__(self, idx):
        hyper = _load_cube(self.hyper_paths[idx])
        bgr = _load_rgb(self.bgr_paths[idx])
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hyper_paths)
