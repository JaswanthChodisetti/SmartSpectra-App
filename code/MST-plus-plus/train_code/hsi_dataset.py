from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import os


class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        # SmartSpectra: Agro-HSR scenes are 512x512 (not the 482x512 of the
        # original NTIRE/ARAD-1K pipeline this repo was written for).
        # The 482-tall hardcode below caused `crop_size + h_idx*stride` to
        # exceed the 512-tall image and raise IndexError during training.
        h, w = 512, 512  # Agro-HSR img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        # SmartSpectra patch: search both Train_Spec/ AND Test_Spec/ (and
        # Train_RGB/, Test_RGB/) when resolving a stem from split_txt. The
        # prep_agrohsr_split.py shuffles all 1322 pairs into train/val
        # regardless of whether they came from Agro-HSR's Train_* or Test_*
        # directory, so a stem "Test_42" can legitimately be in the train
        # split. Original MST++ assumes all train stems live in Train_Spec/.
        hyper_dirs = [
            f'{data_root}/Train_Spec/',
            f'{data_root}/Test_Spec/',
        ]
        bgr_dirs = [
            f'{data_root}/Train_RGB/',
            f'{data_root}/Test_RGB/',
        ]

        def _resolve(stem_ext: str, candidates: list[str]) -> str:
            for d in candidates:
                candidate = os.path.join(d, stem_ext)
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(
                f"Could not find {stem_ext!r} in any of {candidates}"
            )

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of agro-hsr dataset:{len(hyper_list)}')
        print(f'len(bgr) of agro-hsr dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            stem_ext = hyper_list[i]                       # e.g. "Test_42.mat"
            stem = stem_ext.split('.')[0]                  # e.g. "Test_42"
            hyper_path = _resolve(stem_ext, hyper_dirs)
            bgr_path = _resolve(stem + '.jpg', bgr_dirs)
            with h5py.File(hyper_path, 'r') as mat:
                hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            assert stem == bgr_path.split(os.sep)[-1].split('.')[0], \
                f'Hyper and RGB come from different scenes: {stem} vs {bgr_path}'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,512,512]
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Agro-HSR scene {i} ({stem}) is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        # SmartSpectra patch: same multi-dir resolution as TrainDataset.
        hyper_dirs = [
            f'{data_root}/Train_Spec/',
            f'{data_root}/Test_Spec/',
        ]
        bgr_dirs = [
            f'{data_root}/Train_RGB/',
            f'{data_root}/Test_RGB/',
        ]

        def _resolve(stem_ext: str, candidates: list[str]) -> str:
            for d in candidates:
                candidate = os.path.join(d, stem_ext)
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(
                f"Could not find {stem_ext!r} in any of {candidates}"
            )

        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of agro-hsr dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of agro-hsr dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            stem_ext = hyper_list[i]
            stem = stem_ext.split('.')[0]
            hyper_path = _resolve(stem_ext, hyper_dirs)
            bgr_path = _resolve(stem + '.jpg', bgr_dirs)
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            assert stem == bgr_path.split(os.sep)[-1].split('.')[0], \
                f'Hyper and RGB come from different scenes: {stem} vs {bgr_path}'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,512,512]
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Agro-HSR valid scene {i} ({stem}) is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)