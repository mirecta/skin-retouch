"""GAN training script for ABPN-based face retouching."""

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

from model import ABPN, PatchDiscriminator, RetouchLoss
from data import FFHQRDataset


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(G, D, opt_G, opt_D, scaler_G, scaler_D,
                    sched_G, sched_D, epoch, metrics, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'opt_G_state': opt_G.state_dict(),
        'opt_D_state': opt_D.state_dict(),
        'scaler_G_state': scaler_G.state_dict(),
        'scaler_D_state': scaler_D.state_dict(),
        'sched_G_state': sched_G.state_dict(),
        'sched_D_state': sched_D.state_dict(),
        'metrics': metrics,
        'config': cfg,
    }, path)


def get_progressive_stage(cfg, epoch):
    """Return (patch_size, batch_size) for the current epoch."""
    stages = cfg.get('progressive', None)
    if not stages:
        return cfg['data']['patch_size'], cfg['train'].get('batch_size', 16)
    current = stages[0]
    for stage in stages:
        if epoch >= stage[0]:
            current = stage
    return current[1], current[2]


def build_dataloader(cfg, split, patch_size, batch_size, augment):
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


def evaluate(G, val_dl, criterion, device, writer=None, epoch=None):
    """Validation: compute PSNR, SSIM, avg losses; log images to TensorBoard."""
    G.eval()
    totals = {}
    count = 0

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sample_logged = False

    with torch.no_grad():
        for batch in val_dl:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)
            # dataset mask: [B,3,H,W] — take 1st channel as GT for Dice
            mask_gt = batch['mask'][:, :1, :, :].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                R_0, R_l, M, blends = G(inp)
                _, loss_dict = criterion(R_0, R_l, M, blends, tgt, inp=None,
                                         D_fake_logits=None, mask_gt=mask_gt)

            R0_f = R_0.float().clamp(0, 1)
            tgt_f = tgt.float()

            psnr_metric.update(R0_f, tgt_f)
            ssim_metric.update(R0_f, tgt_f)

            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1

            if writer is not None and epoch is not None and not sample_logged:
                n = min(4, inp.shape[0])
                inp_vis = inp[:n].float()
                out_vis = R0_f[:n]
                tgt_vis = tgt_f[:n]

                # Show B_0 (full-res blend map) amplified for visibility
                B_0 = blends[2][:n].float()
                if B_0.shape[1] == 1:
                    B_0 = B_0.expand(-1, 3, -1, -1)
                blend_vis = (B_0 * 3.0).clamp(0, 1)
                if blend_vis.shape[-2:] != inp_vis.shape[-2:]:
                    blend_vis = F.interpolate(
                        blend_vis, size=inp_vis.shape[-2:],
                        mode='bilinear', align_corners=False
                    )

                rows = torch.cat([inp_vis, out_vis, tgt_vis, blend_vis], dim=0)
                grid = make_grid(rows, nrow=n, padding=2, normalize=False)
                writer.add_image('val/samples', grid, epoch)
                sample_logged = True

    avg_metrics = {k: v / count for k, v in totals.items()}
    avg_metrics['psnr'] = psnr_metric.compute().item()
    avg_metrics['ssim'] = ssim_metric.compute().item()
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train ABPN face retouching GAN')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg['train']
    device = torch.device(cfg['hardware']['device'])

    # --- Models ---
    G = ABPN.from_config(cfg['model']).to(device)
    D = PatchDiscriminator(in_ch=6).to(device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator parameters:     {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # --- Loss ---
    criterion = RetouchLoss(
        lambda_mse=tcfg['lambda_mse'],
        lambda_perc=tcfg['lambda_perc'],
        lambda_adv=tcfg['lambda_adv'],
        lambda_dice=tcfg['lambda_dice'],
        lambda_tv=tcfg['lambda_tv'],
        lambda_aesthetic=tcfg.get('lambda_aesthetic', 0.0),
        aesthetic_metric=tcfg.get('aesthetic_metric', 'clipiqa'),
    ).to(device)

    # --- Optimizers ---
    betas = tuple(tcfg['betas'])
    opt_G = torch.optim.Adam(G.parameters(), lr=tcfg['lr_g'], betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=tcfg['lr_d'], betas=betas)

    # --- Schedulers: step decay ×0.1 after step_epoch ---
    step_epoch = tcfg.get('lr_step_epoch', 100)
    sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=step_epoch, gamma=0.1)
    sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=step_epoch, gamma=0.1)

    # --- AMP scalers (one per backward pass) ---
    scaler_G = GradScaler('cuda')
    scaler_D = GradScaler('cuda')

    # --- Fixed val dataloader (256-crop, batch 4) ---
    _, val_dl = build_dataloader(cfg, 'val', 256, 4, augment=False)

    # --- Logging ---
    os.makedirs(cfg['log']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(cfg['log']['log_dir'])
    writer.add_text('config', f'```yaml\n{yaml.dump(cfg)}\n```', 0)

    # --- Resume ---
    start_epoch = 0
    best_psnr = -float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        opt_G.load_state_dict(ckpt['opt_G_state'])
        opt_D.load_state_dict(ckpt['opt_D_state'])
        scaler_G.load_state_dict(ckpt['scaler_G_state'])
        scaler_D.load_state_dict(ckpt['scaler_D_state'])
        start_epoch = ckpt['epoch'] + 1

        if 'sched_G_state' in ckpt and 'sched_D_state' in ckpt:
            sched_G.load_state_dict(ckpt['sched_G_state'])
            sched_D.load_state_dict(ckpt['sched_D_state'])
        else:
            for _ in range(start_epoch):
                sched_G.step()
                sched_D.step()

        best_psnr = ckpt.get('metrics', {}).get('psnr', -float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']}, "
              f"lr_G={opt_G.param_groups[0]['lr']:.2e}, "
              f"lr_D={opt_D.param_groups[0]['lr']:.2e}")

    # --- EMA skip guard state ---
    ema_loss_G = None
    ema_decay = 0.98
    skip_warmup_batches = 100
    seen_batches = 0

    # --- Training loop ---
    current_patch_size = None
    train_dl = None
    d_steps = tcfg.get('d_steps', 1)

    for epoch in range(start_epoch, tcfg['epochs']):
        patch_size, batch_size = get_progressive_stage(cfg, epoch)
        if patch_size != current_patch_size:
            current_patch_size = patch_size
            print(f"\n>>> Progressive stage: patch_size={patch_size}, "
                  f"batch_size={batch_size}")
            _, train_dl = build_dataloader(cfg, 'train', patch_size, batch_size,
                                           augment=True)

        G.train()
        D.train()

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        batch_count = 0
        skipped_G = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{tcfg['epochs']}")

        for batch in pbar:
            inp     = batch['input'].to(device, non_blocking=True)
            tgt     = batch['target'].to(device, non_blocking=True)
            mask_gt = batch['mask'][:, :1, :, :].to(device, non_blocking=True)

            # ----------------------------------------------------------------
            # Discriminator step(s)
            # ----------------------------------------------------------------
            for _ in range(d_steps):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        R_0_d, _, _, _ = G(inp)
                    D_real = D(torch.cat([inp, tgt], dim=1))
                    D_fake = D(torch.cat([inp, R_0_d.detach()], dim=1))
                    loss_D = (
                        0.5 * (D_real - 1.0).pow(2).mean()
                        + 0.5 * D_fake.pow(2).mean()
                    )

                opt_D.zero_grad()
                scaler_D.scale(loss_D).backward()
                scaler_D.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)
                scaler_D.step(opt_D)
                scaler_D.update()

            # ----------------------------------------------------------------
            # Generator step — with EMA skip guard
            # ----------------------------------------------------------------
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                R_0, R_l, M, blends = G(inp)
                D_fake_for_G = D(torch.cat([inp, R_0], dim=1))
                loss_G, loss_dict = criterion(R_0, R_l, M, blends, tgt, inp=inp,
                                              D_fake_logits=D_fake_for_G,
                                              mask_gt=mask_gt)

            total_G = loss_dict['total']
            is_spike = (
                not torch.isfinite(loss_G)
                or (seen_batches >= skip_warmup_batches
                    and ema_loss_G is not None
                    and total_G > 5.0 * ema_loss_G)
            )

            if not is_spike:
                opt_G.zero_grad()
                scaler_G.scale(loss_G).backward()
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                scaler_G.step(opt_G)
                scaler_G.update()

                epoch_loss_G += total_G
                ema_loss_G = (total_G if ema_loss_G is None
                              else ema_decay * ema_loss_G + (1 - ema_decay) * total_G)
            else:
                opt_G.zero_grad()
                skipped_G += 1

            epoch_loss_D += loss_D.item()
            batch_count += 1
            seen_batches += 1

            B0_mean = blends[2].mean().item() if blends else 0.0
            pbar.set_postfix({
                'loss_G': f"{total_G:.4f}",
                'loss_D': f"{loss_D.item():.4f}",
                'mse':    f"{loss_dict['mse']:.4f}",
                'perc':   f"{loss_dict['perc']:.4f}",
                'aes':    f"{loss_dict['aesthetic']:.4f}",
                'ema':    f"{ema_loss_G:.4f}" if ema_loss_G else "n/a",
                'B0':     f"{B0_mean:.3f}",
                'crop':   patch_size,
            })

        sched_G.step()
        sched_D.step()

        avg_G = epoch_loss_G / max(batch_count, 1)
        avg_D = epoch_loss_D / max(batch_count, 1)
        writer.add_scalar('train/loss_G', avg_G, epoch)
        writer.add_scalar('train/loss_D', avg_D, epoch)
        writer.add_scalar('train/mse',    loss_dict['mse'], epoch)
        writer.add_scalar('train/lr_G',   opt_G.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/lr_D',   opt_D.param_groups[0]['lr'], epoch)
        if skipped_G:
            print(f"  [epoch {epoch}] skipped {skipped_G} G updates (spike guard)")

        # --- Validation ---
        if epoch % cfg['log']['eval_every'] == 0:
            val_metrics = evaluate(G, val_dl, criterion, device,
                                   writer=writer, epoch=epoch)

            print(f"  Val — psnr: {val_metrics['psnr']:.2f}  "
                  f"ssim: {val_metrics['ssim']:.4f}  "
                  f"total: {val_metrics['total']:.4f}  "
                  f"mse: {val_metrics['mse']:.4f}  "
                  f"perc: {val_metrics['perc']:.4f}")

            writer.add_scalar('val/psnr',  val_metrics['psnr'],  epoch)
            writer.add_scalar('val/ssim',  val_metrics['ssim'],  epoch)
            writer.add_scalar('val/total', val_metrics['total'], epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    G, D, opt_G, opt_D, scaler_G, scaler_D, sched_G, sched_D,
                    epoch, val_metrics, cfg,
                    os.path.join(cfg['log']['checkpoint_dir'], 'best.pth'),
                )
                print(f"  Saved best checkpoint (psnr={best_psnr:.2f})")

            G.train()

        # --- Periodic checkpoint ---
        if epoch % cfg['log']['save_every'] == 0:
            save_checkpoint(
                G, D, opt_G, opt_D, scaler_G, scaler_D, sched_G, sched_D,
                epoch, {}, cfg,
                os.path.join(cfg['log']['checkpoint_dir'], f'epoch_{epoch:03d}.pth'),
            )

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
