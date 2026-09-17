from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.reshape(-1))
        return mrae


class Loss_MRAE_custom(nn.Module):
    def __init__(self):
        super(Loss_MRAE_custom, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        mask = label == 0
        if mask.any():
            label_wo_zero = label.clone()
            label_wo_zero[mask] = 1e-8
        else:
            label_wo_zero = label
        error = torch.abs(outputs - label) / label_wo_zero
        mrae = torch.mean(error)
        return mrae


class Loss_L1(nn.Module):
    """SmartSpectra patch: plain L1 loss as the dominant reconstruction term.

    Why we replaced Loss_MRAE: see the project note "Problems Encountered /
    Problem 3". Agro-HSR labels are sparse (mean=0.03–0.11, p50=0.00, max=1.0).
    MRAE = |out - label| / label explodes to 1e7+ on near-zero labels, and
    Loss_MRAE_custom's divisor floor (1e-8) doesn't help — it just changes
    inf→1e7 while still dominating the smoothness regularizers by 5+ orders
    of magnitude. The model can drive this loss toward zero by predicting
    near-zero everywhere, which is the exact failure mode that collapsed
    the original training.

    L1 in absolute units is bounded by the label range [0, 1], so the
    smoothness/spectral regularizers (lambda=0.02) actually contribute
    meaningfully to the gradient. Combined with the sigmoid output clamp
    (MST_Plus_Plus.use_sigmoid=True), L1 alone is enough.
    """
    def __init__(self):
        super(Loss_L1, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        return torch.mean(torch.abs(outputs - label))


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

class Loss_Smoothness(nn.Module):
    """Spatial total-variation penalty on the predicted HSI cube.
    Penalizes |dx| + |dy| across the (B, C, H, W) output. Keeps reflectance
    smooth without smoothing across spectral channels.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred):
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        return (dx.mean() + dy.mean()) / 2.0


class Loss_SpectralSmoothness(nn.Module):
    """Spectral total-variation penalty across adjacent wavelength bands.

    Unlike Loss_Smoothness (which is spatial), this penalizes
    |pred[:, :, b+1] - pred[:, :, b]| across the C (band) axis. It enforces
    that reconstructed spectra are piecewise-smooth in wavelength — which is
    physically true for natural reflectance (no sharp spectrometer-resolution
    features in the 400–1000 nm range) and which prevents the model from
    learning band-wise noise as a shortcut.

    The 2D shape (H, W) per band is preserved; we just diff along the channel
    axis. Average is over (B, H, W, C-1) so the loss is scale-equivalent to
    Loss_Smoothness.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred):
        # pred: (B, C, H, W). Diff along channel axis.
        dlam = torch.abs(pred[:, 1:, :, :] - pred[:, :-1, :, :])
        return dlam.mean()

class Loss_NonNeg(nn.Module):
    """Soft penalty for negative reflectance values. ReLU on the
    negative half so a small negative is mildly punished but doesn't
    explode the gradient.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred):
        return torch.relu(-pred).mean()

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
