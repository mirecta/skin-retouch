"""Evaluation script — PSNR / SSIM / LPIPS / blend_mean on test split."""

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ABPN
from data import FFHQRDataset


def compute_psnr(pred, target):
    """PSNR between two [0, 1] tensors."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, target):
    """SSIM using skimage (per-image, then average)."""
    from skimage.metrics import structural_similarity
    p = pred.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    t = target.cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(p, t, channel_axis=2, data_range=1.0)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ABPN on FFHQR test split')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML (if not embedded in checkpoint)')
    parser.add_argument('--original-dir', type=str, default=None)
    parser.add_argument('--retouched-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--tensorboard-dir', type=str, default=None,
                        help='Directory for TensorBoard logs (optional)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Config: CLI flag takes priority, then checkpoint, then error
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt.get('config', None)
        if cfg is None:
            raise ValueError("No config embedded in checkpoint; provide --config")

    # Model — ABPN forward: (R_0, R_l, M, blends) = model(x)
    model = ABPN.from_config(cfg['model']).to(device)
    state_key = 'G_state' if 'G_state' in ckpt else 'model_state'
    model.load_state_dict(ckpt[state_key])
    model.eval()

    # Data
    original_dir = args.original_dir or cfg['data']['original_dir']
    retouched_dir = args.retouched_dir or cfg['data']['retouched_dir']

    test_ds = FFHQRDataset(
        original_dir=original_dir,
        retouched_dir=retouched_dir,
        split='test',
        augment=False,
    )
    test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4)

    # Optional: LPIPS
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        use_lpips = True
    except ImportError:
        print("Warning: lpips not installed, skipping LPIPS metric")
        use_lpips = False

    # Optional: TensorBoard
    writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.tensorboard_dir)
        except ImportError:
            print("Warning: tensorboard not available, skipping TensorBoard logging")

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    blend_means = []

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dl, desc='Evaluating')):
            inp = batch['input'].to(device)   # [B, 3, H, W]
            tgt = batch['target'].to(device)  # [B, 3, H, W]

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                R_0, R_l, M, blends = model(inp)

            out   = R_0.float().clamp(0, 1)
            tgt   = tgt.float()
            blend = blends[2].float()  # B_0: full-res blend map

            # Blend mean for this batch
            batch_blend_mean = blend.mean().item()
            blend_means.append(batch_blend_mean)

            for i in range(out.shape[0]):
                psnr_vals.append(compute_psnr(out[i], tgt[i]))
                ssim_vals.append(compute_ssim(out[i], tgt[i]))

            if use_lpips:
                # LPIPS expects [-1, 1] range
                lp = lpips_fn(out * 2 - 1, tgt * 2 - 1)
                lpips_vals.extend(
                    lp.squeeze().cpu().tolist() if lp.dim() > 0 else [lp.item()]
                )

            # TensorBoard: log per-batch scalars
            if writer is not None:
                step = batch_idx
                writer.add_scalar('eval/psnr_batch',
                                  np.mean(psnr_vals[-out.shape[0]:]), step)
                writer.add_scalar('eval/blend_mean_batch', batch_blend_mean, step)
                if lpips_vals:
                    writer.add_scalar('eval/lpips_batch',
                                      np.mean(lpips_vals[-out.shape[0]:]), step)

    # Summary
    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    mean_blend = np.mean(blend_means)

    print(f"\n{'='*40}")
    print(f"Test Results ({len(psnr_vals)} images)")
    print(f"{'='*40}")
    print(f"PSNR:       {mean_psnr:.2f} dB  (target: >47 dB)")
    print(f"SSIM:       {mean_ssim:.4f}     (target: >0.990)")
    if lpips_vals:
        mean_lpips = np.mean(lpips_vals)
        print(f"LPIPS:      {mean_lpips:.4f}     (target: <0.013)")
    print(f"Blend mean: {mean_blend:.4f}")

    # TensorBoard: log final summary scalars
    if writer is not None:
        writer.add_scalar('eval/psnr_final', mean_psnr, 0)
        writer.add_scalar('eval/ssim_final', mean_ssim, 0)
        writer.add_scalar('eval/blend_mean_final', mean_blend, 0)
        if lpips_vals:
            writer.add_scalar('eval/lpips_final', np.mean(lpips_vals), 0)
        writer.close()
        print(f"\nTensorBoard logs written to: {args.tensorboard_dir}")


if __name__ == '__main__':
    main()
