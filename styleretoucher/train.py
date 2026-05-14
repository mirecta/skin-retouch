"""StyleRetoucher training — train SE + BAFS while keeping StyleGAN2 frozen."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

# Suppress noisy "Setting up PyTorch plugin ... Failed!" warnings from stylegan3
import warnings
warnings.filterwarnings('ignore')

# Make parent package importable
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from styleretoucher import StyleRetoucher
from data import FFHQRDataset


# ---------------------------------------------------------------------------
# Losses — minimal set for StyleRetoucher
# ---------------------------------------------------------------------------

import torchvision.models as models


class _PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg[:9]))    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg[9:18]))  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, output, target):
        out_n = (output - self.mean) / self.std
        tgt_n = (target - self.mean) / self.std
        f1_o = self.slice1(out_n);  f1_t = self.slice1(tgt_n)
        f2_o = self.slice2(f1_o);   f2_t = self.slice2(f1_t)
        return F.l1_loss(f1_o, f1_t) + F.l1_loss(f2_o, f2_t)


class _IdentityLoss(nn.Module):
    """VGG16-based identity preservation: cosine distance on a face crop."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.encoder = nn.Sequential(*list(vgg[:23]))  # relu4_2
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _embed(self, x):
        # Centre crop to ~half the image height, normalise
        h, w = x.shape[-2:]
        cy, cx = h // 2, w // 2
        s = min(h, w) // 4
        x = x[:, :, cy - s:cy + s, cx - s:cx + s]
        x = (x - self.mean) / self.std
        f = self.encoder(x).flatten(1)
        return F.normalize(f, dim=1)

    def forward(self, output, source):
        # Identity preservation: output should be close to *source* (input),
        # not to target — we want to keep the same person.
        e_o = self._embed(output)
        e_s = self._embed(source)
        return 1.0 - (e_o * e_s).sum(dim=1).mean()


class StyleRetoucherLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_perc=0.5, lambda_identity=0.1):
        super().__init__()
        self.lambda_mse      = lambda_mse
        self.lambda_perc     = lambda_perc
        self.lambda_identity = lambda_identity
        self.perc_fn     = _PerceptualLoss()
        self.identity_fn = _IdentityLoss()

    def forward(self, output, target, source):
        mse  = F.mse_loss(output, target)
        perc = self.perc_fn(output.float(), target.float())
        ide  = self.identity_fn(output.float(), source.float())
        total = self.lambda_mse * mse + self.lambda_perc * perc + self.lambda_identity * ide
        return total, {
            'mse':      mse.item(),
            'perc':     perc.item(),
            'identity': ide.item(),
            'total':    total.item(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, metrics, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':     epoch,
        'se_state':  model.se.state_dict(),
        'bafs_state': model.bafs.state_dict(),
        'opt_state': optimizer.state_dict(),
        'metrics':   metrics,
        'config':    cfg,
    }, path)


def build_dataloader(cfg, split, augment):
    ds = FFHQRDataset(
        original_dir=cfg['data']['original_dir'],
        retouched_dir=cfg['data']['retouched_dir'],
        split=split,
        patch_size=1024,           # full FFHQR resolution — StyleGAN2 fixed at 1024
        augment=augment,
        limit=cfg['data'].get('limit_' + split, None),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    return ds, dl


def evaluate(model, val_dl, criterion, device, writer=None, epoch=None):
    model.eval()
    totals = {}
    count = 0

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sample_logged = False

    with torch.no_grad():
        for batch in val_dl:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)

            out = model(inp)
            _, loss_dict = criterion(out, tgt, inp)

            psnr_metric.update(out, tgt)
            ssim_metric.update(out, tgt)
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1

            if writer is not None and epoch is not None and not sample_logged:
                n = min(2, inp.shape[0])
                rows = torch.cat([inp[:n], out[:n], tgt[:n]], dim=3)
                grid = make_grid(rows, nrow=1, padding=2, normalize=False)
                writer.add_image('val/samples', grid, epoch)
                sample_logged = True

    avg = {k: v / count for k, v in totals.items()}
    avg['psnr'] = psnr_metric.compute().item()
    avg['ssim'] = ssim_metric.compute().item()
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train StyleRetoucher on FFHQR')
    parser.add_argument('--config', type=str, default='styleretoucher/configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg['train']
    device = torch.device(cfg['hardware']['device'])

    # --- Model ---
    sg2_pkl = cfg['model']['stylegan2_pkl']
    model = StyleRetoucher(stylegan2_pkl=sg2_pkl).to(device)

    trainable = sum(p.numel() for p in model.trainable_parameters if p.requires_grad)
    frozen    = sum(p.numel() for p in model.sg2.parameters())
    print(f'Trainable params (SE+BAFS): {trainable:,}')
    print(f'Frozen params (StyleGAN2):  {frozen:,}')

    # --- Loss ---
    criterion = StyleRetoucherLoss(
        lambda_mse=tcfg['lambda_mse'],
        lambda_perc=tcfg['lambda_perc'],
        lambda_identity=tcfg['lambda_identity'],
    ).to(device)

    # --- Optimizer (only SE + BAFS train) ---
    opt = torch.optim.Adam(
        [p for p in model.trainable_parameters if p.requires_grad],
        lr=tcfg['lr'],
        betas=tuple(tcfg['betas']),
    )

    # --- Data ---
    _, train_dl = build_dataloader(cfg, 'train', augment=True)
    _, val_dl   = build_dataloader(cfg, 'val',   augment=False)

    # --- Logging ---
    os.makedirs(cfg['log']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(cfg['log']['log_dir'])
    writer.add_text('config', f'```yaml\n{yaml.dump(cfg)}\n```', 0)

    # --- Resume ---
    start_epoch = 0
    best_psnr = -float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.se.load_state_dict(ckpt['se_state'])
        model.bafs.load_state_dict(ckpt['bafs_state'])
        opt.load_state_dict(ckpt['opt_state'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('metrics', {}).get('psnr', -float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']}, lr={opt.param_groups[0]['lr']:.2e}")

    # --- Training loop ---
    for epoch in range(start_epoch, tcfg['epochs']):
        model.train()
        # Keep frozen StyleGAN2 in eval mode (BatchNorm-free, but safer)
        model.sg2.eval()

        epoch_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{tcfg['epochs']}")
        for batch in pbar:
            inp = batch['input'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)

            out = model(inp)
            loss, ld = criterion(out, tgt, inp)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.trainable_parameters if p.requires_grad],
                max_norm=1.0,
            )
            opt.step()

            epoch_loss += ld['total']
            batch_count += 1
            pbar.set_postfix({
                'loss':  f"{ld['total']:.4f}",
                'mse':   f"{ld['mse']:.4f}",
                'perc':  f"{ld['perc']:.4f}",
                'ident': f"{ld['identity']:.4f}",
            })

        avg = epoch_loss / max(batch_count, 1)
        writer.add_scalar('train/loss', avg, epoch)

        # --- Validation ---
        if epoch % cfg['log']['eval_every'] == 0:
            val_metrics = evaluate(model, val_dl, criterion, device,
                                   writer=writer, epoch=epoch)
            print(f"  Val — psnr: {val_metrics['psnr']:.2f}  "
                  f"ssim: {val_metrics['ssim']:.4f}  "
                  f"total: {val_metrics['total']:.4f}")

            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    model, opt, epoch, val_metrics, cfg,
                    os.path.join(cfg['log']['checkpoint_dir'], 'best.pth'),
                )
                print(f"  Saved best (psnr={best_psnr:.2f})")

        # --- Periodic checkpoint ---
        if epoch % cfg['log']['save_every'] == 0:
            save_checkpoint(
                model, opt, epoch, {}, cfg,
                os.path.join(cfg['log']['checkpoint_dir'], f'epoch_{epoch:03d}.pth'),
            )

    writer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
