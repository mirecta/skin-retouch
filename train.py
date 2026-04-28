"""Training script for FaceRetouchNet."""

import math
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

from model import FaceRetouchNet, RetouchLoss
from data import FFHQRDataset


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, metrics, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'metrics': metrics,
        'config': cfg,
    }, path)


def evaluate(model, val_dl, criterion, device, writer=None, epoch=None):
    """Run validation and return average metrics + log images/metrics to TensorBoard."""
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
            mgt = batch['mask'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out, mpred = model(inp)
                _, loss_dict = criterion(out, mpred, tgt, mgt, source=inp)

            out_f = out.float().clamp(0, 1)
            tgt_f = tgt.float()

            psnr_metric.update(out_f, tgt_f)
            ssim_metric.update(out_f, tgt_f)

            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1

            # Log sample images from the first batch
            if writer is not None and epoch is not None and not sample_logged:
                n = min(4, inp.shape[0])
                inp_f = inp[:n].float()
                out_vis = out_f[:n]
                tgt_vis = tgt_f[:n]

                # Amplified diff (10x) between output and input
                diff = (out_vis - inp_f).abs() * 10.0
                diff = diff.clamp(0, 1)

                # Mask visualization (upsample to match)
                mask_vis = mpred[:n].float().mean(dim=1, keepdim=True)
                mask_vis = F.interpolate(mask_vis, size=inp_f.shape[-2:],
                                         mode='bilinear', align_corners=False)
                mask_vis = mask_vis.expand(-1, 3, -1, -1)

                # Row: input | output | target | diff_10x | mask
                rows = torch.cat([inp_f, out_vis, tgt_vis, diff, mask_vis], dim=0)
                grid = make_grid(rows, nrow=n, padding=2, normalize=False)
                writer.add_image('val/samples', grid, epoch)

                sample_logged = True

    avg_metrics = {k: v / count for k, v in totals.items()}
    avg_metrics['psnr'] = psnr_metric.compute().item()
    avg_metrics['ssim'] = ssim_metric.compute().item()

    return avg_metrics


def get_progressive_stage(cfg, epoch):
    """Get patch_size and batch_size for current epoch from progressive schedule."""
    stages = cfg.get('progressive', cfg['data'].get('progressive', None))
    if not stages:
        return cfg['data']['patch_size'], cfg['train']['batch_size']
    # Find the latest stage that has started
    current = stages[0]
    for stage in stages:
        if epoch >= stage[0]:
            current = stage
    return current[1], current[2]


def build_dataloader(cfg, split, patch_size, batch_size, augment):
    """Build dataset + dataloader with given patch/batch size."""
    ds = FFHQRDataset(
        original_dir=cfg['data']['original_dir'],
        retouched_dir=cfg['data']['retouched_dir'],
        split=split,
        patch_size=patch_size,
        augment=augment,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        drop_last=(split == 'train'),
    )
    return ds, dl


def main():
    parser = argparse.ArgumentParser(description='Train FaceRetouchNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg['hardware']['device'])

    # Model
    model = FaceRetouchNet.from_config(cfg['model']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        betas=tuple(cfg['train']['betas']),
        weight_decay=1e-4,
    )

    # Cosine annealing with linear warmup
    warmup_epochs = cfg['train'].get('warmup_epochs', 5)
    total_epochs = cfg['train']['epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # linear warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    # Loss
    criterion = RetouchLoss(
        lambda_mask=cfg['loss']['lambda_mask'],
        lambda_perceptual=cfg['loss']['lambda_perceptual'],
    ).to(device)

    # Validation dataloader (fixed, 512 crops to avoid OOM at full 1024)
    _, val_dl = build_dataloader(cfg, 'val', 512, 4, augment=False)

    # Logging
    os.makedirs(cfg['log']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(cfg['log']['log_dir'])

    # Log hyperparameters
    writer.add_text('config', f'```yaml\n{yaml.dump(cfg)}\n```', 0)

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = ckpt['epoch'] + 1
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        else:
            # Older checkpoint without scheduler state — fast-forward by
            # stepping scheduler to match start_epoch so LR is correct.
            for _ in range(start_epoch):
                scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}, lr={optimizer.param_groups[0]['lr']:.2e}")

    # Training loop
    current_patch_size = None
    train_dl = None

    # EMA of recent batch losses — survives across epochs so the skip guard
    # has a stable reference even when a whole epoch gets skipped.
    ema_loss = None
    ema_decay = 0.98
    skip_warmup_batches = 100  # disable guard for first N batches after start/resume
    seen_batches = 0

    for epoch in range(start_epoch, cfg['train']['epochs']):
        # Progressive training: rebuild dataloader if stage changed
        patch_size, batch_size = get_progressive_stage(cfg, epoch)
        if patch_size != current_patch_size:
            current_patch_size = patch_size
            print(f"\n>>> Progressive stage: patch_size={patch_size}, "
                  f"batch_size={batch_size}")
            _, train_dl = build_dataloader(cfg, 'train', patch_size, batch_size,
                                           augment=True)

        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        epoch_loss = 0.0
        batch_count = 0
        skipped = 0

        for batch in pbar:
            inp = batch['input'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)
            mgt = batch['mask'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out, mpred = model(inp)
                loss, loss_dict = criterion(out, mpred, tgt, mgt, source=inp)

            # Skip anomalous loss spikes (NaN/Inf or >5x EMA after warmup).
            # Guard disabled during warmup so EMA can establish.
            total = loss_dict['total']
            is_spike = (
                not torch.isfinite(loss)
                or (seen_batches >= skip_warmup_batches
                    and ema_loss is not None
                    and total > 5.0 * ema_loss)
            )
            if is_spike:
                optimizer.zero_grad()
                skipped += 1
                pbar.set_postfix({'SKIPPED': skipped, 'loss': f"{total:.4f}",
                                  'ema': f"{ema_loss:.4f}" if ema_loss else "n/a"})
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total
            batch_count += 1
            seen_batches += 1
            ema_loss = total if ema_loss is None else ema_decay * ema_loss + (1 - ema_decay) * total
            pbar.set_postfix({
                'loss': f"{total:.4f}",
                'ema': f"{ema_loss:.4f}",
                'l1': f"{loss_dict['l_retouch']:.4f}",
                'mask': f"{loss_dict['l_mask']:.4f}",
                'crop': patch_size,
            })

        scheduler.step()

        avg_train_loss = epoch_loss / max(batch_count, 1)
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Validation
        if epoch % cfg['log']['eval_every'] == 0:
            val_metrics = evaluate(model, val_dl, criterion, device,
                                   writer=writer, epoch=epoch)
            print(f"  Val — total: {val_metrics['total']:.4f}, "
                  f"l1: {val_metrics['l_retouch']:.4f}, "
                  f"mask: {val_metrics['l_mask']:.4f}, "
                  f"psnr: {val_metrics['psnr']:.2f}, "
                  f"ssim: {val_metrics['ssim']:.4f}")

            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            # Save best
            if val_metrics['total'] < best_loss:
                best_loss = val_metrics['total']
                save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_metrics, cfg,
                                os.path.join(cfg['log']['checkpoint_dir'], 'best.pth'))
                print(f"  Saved best checkpoint (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % cfg['log']['save_every'] == 0:
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, {},  cfg,
                            os.path.join(cfg['log']['checkpoint_dir'],
                                         f'epoch_{epoch:03d}.pth'))

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
