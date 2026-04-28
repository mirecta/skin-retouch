"""Evaluation script — PSNR / SSIM / LPIPS on test split."""

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FaceRetouchNet
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
    parser = argparse.ArgumentParser(description='Evaluate FaceRetouchNet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Config file (if not embedded in checkpoint)')
    parser.add_argument('--original-dir', type=str, default=None)
    parser.add_argument('--retouched-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Config from checkpoint or file
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt.get('config', None)
        if cfg is None:
            raise ValueError("No config in checkpoint; provide --config")

    # Model
    model = FaceRetouchNet.from_config(cfg['model']).to(device)
    model.load_state_dict(ckpt['model_state'])
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

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc='Evaluating'):
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out, _ = model(inp)

            out = out.float().clamp(0, 1)
            tgt = tgt.float()

            for i in range(out.shape[0]):
                psnr_vals.append(compute_psnr(out[i], tgt[i]))
                ssim_vals.append(compute_ssim(out[i], tgt[i]))

            if use_lpips:
                # LPIPS expects [-1, 1] range
                lp = lpips_fn(out * 2 - 1, tgt * 2 - 1)
                lpips_vals.extend(lp.squeeze().cpu().tolist() if lp.dim() > 0
                                  else [lp.item()])

    print(f"\n{'='*40}")
    print(f"Test Results ({len(psnr_vals)} images)")
    print(f"{'='*40}")
    print(f"PSNR:  {np.mean(psnr_vals):.2f} dB  (target: >47 dB)")
    print(f"SSIM:  {np.mean(ssim_vals):.4f}     (target: >0.990)")
    if lpips_vals:
        print(f"LPIPS: {np.mean(lpips_vals):.4f}     (target: <0.013)")


if __name__ == '__main__':
    main()
