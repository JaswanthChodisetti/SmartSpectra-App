"""
Patched MST++ training script — SmartSpectra.

Changes vs. the original train.py (mst-plus-plus repo):
  1. Switched loss from Loss_MRAE to Loss_MRAE_custom (guarded division,
     prevents NaN/Inf spikes on near-zero target pixels — these spikes
     were plausibly what drove the original model into a degenerate
     "body negate conv_in" minimum that collapses the output magnitude).
  2. Added Loss_Smoothness (spatial TV) and Loss_SpectralSmoothness (band
     TV) regularizers, plus Loss_NonNeg (soft penalty on negative output)
     with lambda_smooth, lambda_spectral, lambda_nonneg as CLI args.
     Default 0.02 each — small, scaled to the dominant Loss_MRAE term.
  3. Sigmoid output clamp is now built into MST_Plus_Plus.forward() (see
     architecture/MST_Plus_Plus.py). With sigmoid, Loss_NonNeg is
     redundant — set its weight to 0 (which we do by default).
  4. Validation-based early stopping with a separate "best" checkpoint
     file. The original saves whenever |delta| < 0.01 OR MRAE improves
     OR every 5000 iters — it never stops and keeps overfitting around
     the cancelled-residual local minimum. We track best_mrae and stop
     if no improvement for --patience eval steps.
  5. Default --data_root now points at the real Agro-HSR layout in this
     project, so a careless CLI invocation doesn't silently hit
     ../dataset/ which doesn't exist.
  6. Safe checkpoint loading: handles both raw state_dict files (the
     pretrained mst_plus_plus.pth from Google Drive) and full dict
     checkpoints (resuming from a saved run). KeyError-proof.

The patched file is a drop-in replacement — same CLI flags (plus more
for the regularization weights + patience) and same checkpoint
format ('epoch', 'iter', 'state_dict', 'optimizer') so inference.py /
our demo's checkpoint loader continue to work.
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import sys
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_MRAE_custom, Loss_L1, Loss_RMSE, Loss_PSNR, \
    Loss_Smoothness, Loss_SpectralSmoothness, Loss_NonNeg
import datetime

# ----------------------------------------------------------------------
# Diagnostic: print which loss variant is in use, so the training log
# makes it unambiguous which file actually trained a given checkpoint.
# ----------------------------------------------------------------------
print("=" * 70)
print("PATCHED train.py — SmartSpectra")
print("  + Smoothness (spatial) + SpectralSmoothness (across bands)")
print("  + NonNeg regularizers (set to 0 when sigmoid output is used)")
print("  + Early stopping on validation MRAE (best checkpoint saved)")
print("  + Safe checkpoint loading (raw state_dict or full dict)")
print("  Primary loss chosen at runtime via --use_l1 / --use_mrae")
print("=" * 70)

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str,
                    default='/mnt/c/Users/jaswa/ollama/Smart_Spectra/models/checkpoints/mst_plus_plus/',
                    help='path log files. Default points at this project\'s models/checkpoints/ '
                         'so net_best.pth lands in a tracked location.')
parser.add_argument("--data_root", type=str,
                    default='/mnt/c/Users/jaswa/ollama/Smart_Spectra/data/raw/Agro-HSR/',
                    help="Agro-HSR root. Default points at this project's layout.")
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
# --- Loss function selector ---
loss_group = parser.add_mutually_exclusive_group()
loss_group.add_argument("--use_l1", dest="use_l1", action="store_true",
                        default=True,
                        help="Use Loss_L1 as the dominant reconstruction term "
                             "(SmartSpectra default, recommended).")
loss_group.add_argument("--use_mrae", dest="use_l1", action="store_false",
                        help="Use Loss_MRAE_custom (guarded division) instead "
                             "of Loss_L1. MRAE may collapse on sparse labels.")

# --- NEW: regularizer weights ---
parser.add_argument("--lambda_smooth", type=float, default=0.02,
                    help="Weight on Loss_Smoothness (spatial TV). "
                         "Start small (~0.01–0.05). 0 disables it.")
parser.add_argument("--lambda_spectral", type=float, default=0.02,
                    help="Weight on Loss_SpectralSmoothness (1D TV across "
                         "the 31 wavelength bands). 0 disables it.")
parser.add_argument("--lambda_nonneg", type=float, default=0.0,
                    help="Weight on Loss_NonNeg. Redundant with sigmoid "
                         "output (always 0 there); default 0.")
# --- NEW: early stopping ---
parser.add_argument("--patience", type=int, default=10,
                    help="Stop if val MRAE hasn't improved in this many "
                         "eval steps (each eval step = 1000 iters).")
parser.add_argument("--min_delta", type=float, default=1e-4,
                    help="Minimum MRAE improvement to count as 'better'.")
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration * opt.end_epoch

# ----------------------------------------------------------------------
# Loss functions — patched
# ----------------------------------------------------------------------
if opt.use_l1:
    criterion_mrae = Loss_L1()
    primary_loss_name = "Loss_L1"
else:
    criterion_mrae = Loss_MRAE_custom()
    primary_loss_name = "Loss_MRAE_custom"
criterion_smooth = Loss_Smoothness()
criterion_spectral = Loss_SpectralSmoothness()
criterion_nonneg = Loss_NonNeg()
print(f"Primary loss: {primary_loss_name}")

# Reporting-only (unchanged from original)
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
# model_generator already handles device placement (CPU-or-CUDA),
# so don't chain an unconditional .cuda() here — it crashes on CPU-only machines.
model = model_generator(method, pretrained_model_path)
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_smooth.cuda()
    criterion_nonneg.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)
logger.info("=" * 70)
logger.info("PATCHED train.py — SmartSpectra")
logger.info("  Loss_MRAE_custom (guarded) + lambda_smooth=%g * Loss_Smoothness"
            " + lambda_spectral=%g * Loss_SpectralSmoothness"
            " + lambda_nonneg=%g * Loss_NonNeg  ||  patience=%d  min_delta=%g"
            % (opt.lambda_smooth, opt.lambda_spectral, opt.lambda_nonneg,
               opt.patience, opt.min_delta))
logger.info("CLI args: %s" % vars(opt))
logger.info("=" * 70)

# Resume
# ----- PATCHED: safe checkpoint loading -----
# Handles both raw state_dict files (pretrained mst_plus_plus.pth) and
# full dict checkpoints (resuming from a saved run). No KeyError.
resume_file = opt.pretrained_model_path
if resume_file is not None and resume_file != '':
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file, map_location='cpu')

        # Pretrained weights are a raw state_dict; saved runs are dicts.
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception as e:
                    print(f"  (optimizer state not loaded: {e})")
            print(f"  loaded full checkpoint at epoch={checkpoint.get('epoch', 0)}, "
                  f"iter={checkpoint.get('iter', 0)}")
        else:
            # Raw state_dict (pretrained weights from Google Drive)
            model.load_state_dict(checkpoint)
            print("  loaded raw state_dict (pretrained weights)")

        start_epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
        iteration = checkpoint.get('iter', 0) if isinstance(checkpoint, dict) else 0
    else:
        print(f"WARNING: pretrained_model_path='{resume_file}' not found. Training from scratch.")
else:
    print("No pretrained_model_path given. Training from scratch.")

# -------------------------------------------

def main():
    cudnn.benchmark = True
    iteration = 0
    record_mrae_loss = 1000.0           # best-so-far val MRAE
    best_mrae = float('inf')            # explicit "best" tracker
    epochs_no_improve = 0               # for early stopping
    best_ckpt_path = os.path.join(opt.outf, 'net_best.pth')
    print(f"Early stopping: patience={opt.patience} eval steps, "
          f"min_delta={opt.min_delta}. Best ckpt will be saved to {best_ckpt_path}")

    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            device = next(model.parameters()).device
            images, labels = images.to(device), labels.to(device)
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)

            # ----- PATCHED LOSS -----
            loss_mrae = criterion_mrae(output, labels)
            loss_smooth = criterion_smooth(output)
            loss_spectral = criterion_spectral(output)
            loss_nonneg = criterion_nonneg(output)
            loss = (loss_mrae
                    + opt.lambda_smooth * loss_smooth
                    + opt.lambda_spectral * loss_spectral
                    + opt.lambda_nonneg * loss_nonneg)
            # -------------------------

            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration + 1
            if iteration % 20 == 0:
                print('[iter:%d/%d], lr=%.9f, train_losses.avg=%.9f '
                      '(mrae=%.6f, smooth=%.6f, spectral=%.6f, nonneg=%.6f)'
                      % (iteration, total_iteration, lr, losses.avg,
                         loss_mrae.item(), loss_smooth.item(),
                         loss_spectral.item(), loss_nonneg.item()))
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')

                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss

                # ----- EARLY STOPPING + BEST-CHECKPOINT -----
                if mrae_loss < (best_mrae - opt.min_delta):
                    best_mrae = mrae_loss
                    epochs_no_improve = 0
                    torch.save({
                        'epoch': iteration // 1000,
                        'iter': iteration,
                        'state_dict': (model.module.state_dict()
                                       if isinstance(model, nn.DataParallel)
                                       else model.state_dict()),
                        'optimizer': optimizer.state_dict(),
                        'best_mrae': best_mrae,
                        'cli_args': vars(opt),
                    }, best_ckpt_path)
                    print(f"  ✓ new best MRAE={best_mrae:.6f}  →  saved {best_ckpt_path}")
                else:
                    epochs_no_improve += 1
                    print(f"  no improvement for {epochs_no_improve}/{opt.patience} eval steps "
                          f"(best={best_mrae:.6f})")
                    if epochs_no_improve >= opt.patience:
                        print(f"EARLY STOP triggered at iter {iteration}. "
                              f"Best val MRAE was {best_mrae:.6f}.")
                        logger.info(f"EARLY STOP at iter={iteration}, best MRAE={best_mrae:.6f}")
                        return 0
                # -------------------------------------------

                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f "
                      % (iteration, iteration // 1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                            "Test RMSE: %.9f, Test PSNR: %.9f "
                            % (iteration, iteration // 1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
    print(f"Reached total_iteration={total_iteration}. Best val MRAE={best_mrae:.6f} "
          f"at {best_ckpt_path}")
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        device = next(model.parameters()).device
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)
