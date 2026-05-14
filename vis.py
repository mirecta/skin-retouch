"""
Quick visual inspection of ABPN checkpoints.
Saves: original | retouched_by_model | ground_truth  side by side.

Usage:
  python inspect.py                          # best.pth, 16 random test images
  python inspect.py --checkpoint checkpoints/epoch_030.pth
  python inspect.py --n 32 --seed 99
"""

import argparse
import os
import random
import torch
from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode

from model import ABPN
from data import FFHQRDataset


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch = ckpt.get('epoch', '?')
    psnr  = ckpt.get('metrics', {}).get('psnr', '?')
    print(f"Checkpoint: epoch={epoch}  val_psnr={psnr}")

    cfg = ckpt.get('config', {})
    G = ABPN.from_config(cfg.get('model', {})).to(device)
    G.load_state_dict(ckpt['G_state'])
    G.eval()

    ds = FFHQRDataset(
        original_dir=cfg['data']['original_dir'],
        retouched_dir=cfg['data']['retouched_dir'],
        split='test',
        patch_size=512,
        augment=False,
    )

    random.seed(args.seed)
    indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            batch = ds[idx]
            inp = batch['input'].unsqueeze(0).to(device)
            tgt = batch['target'].unsqueeze(0).to(device)
            name = batch['name']

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                R_0, R_l, M, blends = G(inp)

            R_0 = R_0.float().clamp(0, 1)

            # Individual files
            save_image(inp.float(),  os.path.join(args.out_dir, f'{name}_1_orig.png'))
            save_image(R_0,          os.path.join(args.out_dir, f'{name}_2_out_e{epoch}.png'))
            save_image(tgt.float(),  os.path.join(args.out_dir, f'{name}_3_gt.png'))
            # Side-by-side strip
            row = torch.cat([inp.float(), R_0, tgt.float()], dim=3)
            save_image(row, os.path.join(args.out_dir, f'{name}_compare_e{epoch}.png'))
            print(f"  [{i+1}/{len(indices)}] {name}")

    print(f"\nDone. {len(indices)} images in {args.out_dir}/")
    print("Layout: [ original | model output | ground truth ]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best.pth')
    parser.add_argument('--out-dir',    default='output/inspect')
    parser.add_argument('--n',          type=int, default=16)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()
    run(args)
