"""CLI inference script — replaces skin_retouch.py."""

import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import FaceRetouchNet


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = FaceRetouchNet.from_config(cfg['model']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def pad_to_multiple(img, multiple=16):
    """Pad image to be divisible by multiple."""
    _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, [0, pad_w, 0, pad_h], mode='reflect')
    return img, h, w


def tiled_inference(model, image, tile_size=1024, overlap=128, device='cuda'):
    """
    Process large images in overlapping tiles with cosine blending.

    Args:
        model: FaceRetouchNet
        image: Tensor [3, H, W] in [0, 1]
        tile_size: tile dimension
        overlap: overlap between tiles
        device: computation device
    Returns:
        output: Tensor [3, H, W]
        mask:   Tensor [3, H/4, W/4]
    """
    _, H, W = image.shape
    stride = tile_size - overlap

    # Create cosine blend window
    ramp = torch.linspace(0, 1, overlap, device=device)
    ones = torch.ones(tile_size - 2 * overlap, device=device)
    window_1d = torch.cat([ramp, ones, ramp.flip(0)])
    window = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)  # [tile, tile]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, tile, tile]

    output = torch.zeros(1, 3, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Clamp tile coordinates
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = image[:, y_start:y_end, x_start:x_end].unsqueeze(0).to(device)

            # Pad if tile is smaller than tile_size
            th, tw = tile.shape[-2:]
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, [0, tile_size - tw, 0, tile_size - th],
                             mode='reflect')

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out_tile, _ = model(tile)

            out_tile = out_tile.float()[:, :, :th, :tw]

            # Current window (may be smaller at edges)
            w = window[:, :, :th, :tw]

            output[:, :, y_start:y_end, x_start:x_end] += out_tile * w
            weight[:, :, y_start:y_end, x_start:x_end] += w

    output = output / weight.clamp(min=1e-8)

    # Get mask from a center crop pass
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        center_size = min(tile_size, H, W)
        cy, cx = H // 2 - center_size // 2, W // 2 - center_size // 2
        center = image[:, cy:cy + center_size, cx:cx + center_size].unsqueeze(0).to(device)
        center, ch, cw = pad_to_multiple(center.squeeze(0), 16)
        _, mask = model(center.unsqueeze(0))
        mask = mask.float()

    return output.squeeze(0).clamp(0, 1), mask.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description='Face retouching inference')
    parser.add_argument('portrait', type=str, help='Input portrait image')
    parser.add_argument('-o', '--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained .pth checkpoint')
    parser.add_argument('-s', '--strength', type=float, default=1.0,
                        help='Blend factor 0-1 (1 = full model output)')
    parser.add_argument('--tile', type=int, default=1024,
                        help='Tile size for large images')
    parser.add_argument('--mask-only', action='store_true',
                        help='Export masks only')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    # Load image
    img = Image.open(args.portrait).convert('RGB')
    img_tensor = to_tensor(img)  # [3, H, W] in [0, 1]
    _, H, W = img_tensor.shape
    print(f"Input: {args.portrait} ({W}x{H})")

    basename = os.path.splitext(os.path.basename(args.portrait))[0]

    # Inference
    if max(H, W) > args.tile:
        print(f"Using tiled inference (tile={args.tile})")
        output, mask = tiled_inference(model, img_tensor, tile_size=args.tile,
                                       device=device)
    else:
        # Pad to multiple of 16
        img_padded, orig_h, orig_w = pad_to_multiple(img_tensor, 16)
        img_padded = img_padded.unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output, mask = model(img_padded)

        output = output.float().squeeze(0)[:, :orig_h, :orig_w].clamp(0, 1)
        mask = mask.float().squeeze(0)

    # Apply strength blending
    if args.strength < 1.0:
        output = img_tensor.to(device) * (1 - args.strength) + output * args.strength

    # Save outputs
    if not args.mask_only:
        out_path = os.path.join(args.output, f'{basename}_retouched.png')
        to_pil_image(output.cpu()).save(out_path)
        print(f"Saved: {out_path}")

    # Save mask
    mask_path = os.path.join(args.output, f'{basename}_mask.png')
    mask_img = mask.mean(dim=0).cpu()  # [H/4, W/4] grayscale
    mask_img = (mask_img * 255).byte()
    Image.fromarray(mask_img.numpy(), mode='L').save(mask_path)
    print(f"Saved: {mask_path}")


if __name__ == '__main__':
    main()
