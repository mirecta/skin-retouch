"""CLI inference script — ABPN face retouching pipeline.

Pipeline:
  1. Detect face bounding box via MediaPipe → crop with 20% padding
  2. Resize crop to tile_size (square, padded), generate skin mask
  3. Run ABPN on the crop
  4. Frequency-separation paste-back (Strategy C): keep ABPN low-freq + original high-freq
  5. Resize result back, paste into full-resolution image
  6. Tiling fallback when no face is detected or image > tile_size
"""

import argparse
import os
import math
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, gaussian_blur

from model import ABPN


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    """Load ABPN from a checkpoint produced by train.py."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = ABPN.from_config(cfg['model']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Skin mask generation (two-stage)
# ---------------------------------------------------------------------------

def _skin_mask_mediapipe(img_np_rgb):
    """Return [H, W] float32 mask via MediaPipe selfie segmentation (lazy import)."""
    try:
        import mediapipe as mp  # noqa: PLC0415
        SelfieSegmentation = mp.solutions.selfie_segmentation.SelfieSegmentation
        with SelfieSegmentation(model_selection=1) as seg:
            result = seg.process(img_np_rgb)
            if result.segmentation_mask is not None:
                return result.segmentation_mask.astype(np.float32)
    except Exception:
        pass
    return None


def _skin_mask_ycbcr(img_np_rgb):
    """Simple YCbCr skin-tone detection. Returns [H, W] float32 mask."""
    img_pil = Image.fromarray(img_np_rgb).convert('YCbCr')
    ycbcr = np.array(img_pil, dtype=np.float32)
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    mask = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)).astype(np.float32)
    return mask


def _dilate_mask(mask_np, radius=10):
    """Morphological dilation to smooth mask edges. mask_np: [H, W] float32."""
    try:
        import cv2  # noqa: PLC0415
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        # Gaussian blur to soften the edge
        blurred = cv2.GaussianBlur(dilated, (kernel_size, kernel_size), radius / 3.0)
        return blurred.clip(0.0, 1.0)
    except Exception:
        # Fallback: tensor-based dilation approximation via max-pool
        t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        pad = radius
        t = F.max_pool2d(t, kernel_size=2 * radius + 1, stride=1, padding=pad)
        return t.squeeze().numpy().clip(0.0, 1.0)


def generate_skin_mask(img_np_rgb):
    """Two-stage skin mask generation.

    Args:
        img_np_rgb: uint8 numpy [H, W, 3] in RGB order.
    Returns:
        mask: float32 numpy [H, W] in [0, 1].
    """
    mask = _skin_mask_mediapipe(img_np_rgb)
    if mask is None:
        mask = _skin_mask_ycbcr(img_np_rgb)
    mask = _dilate_mask(mask, radius=10)
    return mask


# ---------------------------------------------------------------------------
# Face detection / crop helpers
# ---------------------------------------------------------------------------

def detect_face_box(img_np_rgb):
    """Detect first face bounding box via MediaPipe.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, or None if no face found.
    """
    try:
        import mediapipe as mp  # noqa: PLC0415
        FaceDetection = mp.solutions.face_detection.FaceDetection
        H, W = img_np_rgb.shape[:2]
        with FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
            result = fd.process(img_np_rgb)
            if result.detections:
                det = result.detections[0]
                bb = det.location_data.relative_bounding_box
                x1 = int(bb.xmin * W)
                y1 = int(bb.ymin * H)
                x2 = int((bb.xmin + bb.width) * W)
                y2 = int((bb.ymin + bb.height) * H)
                return x1, y1, x2, y2
    except Exception:
        pass
    return None


def padded_face_box(box, H, W, padding_frac=0.20):
    """Add padding_frac around a bounding box, clamped to image bounds.

    Args:
        box: (x1, y1, x2, y2)
        H, W: image height and width
        padding_frac: fraction of box dimension to add on each side
    Returns:
        (x1, y1, x2, y2) clamped
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)
    return x1, y1, x2, y2


def resize_to_square(img_tensor, size):
    """Resize [C, H, W] tensor to (size, size), keeping aspect and padding.

    Returns:
        resized: [C, size, size]
        pad_info: (pad_top, pad_left, orig_h, orig_w) needed to undo
    """
    _, H, W = img_tensor.shape
    scale = size / max(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    resized = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    pad_bottom = size - new_h - pad_top
    pad_right = size - new_w - pad_left

    resized = F.pad(resized, [pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
    return resized, (pad_top, pad_left, new_h, new_w)


def unpad_and_resize(tensor, pad_info, target_h, target_w):
    """Undo square padding and resize back to (target_h, target_w).

    Args:
        tensor: [C, size, size]
        pad_info: (pad_top, pad_left, new_h, new_w) from resize_to_square
        target_h, target_w: original crop resolution
    """
    pad_top, pad_left, new_h, new_w = pad_info
    # Remove padding
    cropped = tensor[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    # Resize to original
    out = F.interpolate(
        cropped.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Frequency-separation paste-back (Strategy C)
# ---------------------------------------------------------------------------

def freq_sep_pasteback(original_crop, retouched_resized, sigma=2.0):
    """Keep ABPN's low-freq (smoothing/color) + original's high-freq (pores/texture).

    Args:
        original_crop:    [C, H, W] float tensor at original crop resolution, [0, 1]
        retouched_resized:[C, H, W] float tensor upsampled to same resolution, [0, 1]
        sigma:            Gaussian blur sigma for frequency separation
    Returns:
        blended:          [C, H, W] float tensor in [0, 1]
    """
    # Kernel size must be odd and >= 1; use ceil(6*sigma)|1 per convention
    ksize = int(math.ceil(6 * sigma)) | 1
    ksize = max(ksize, 3)

    # Low-frequency layers (batch dim required by gaussian_blur)
    low_orig = gaussian_blur(original_crop, kernel_size=ksize, sigma=sigma)
    low_reto = gaussian_blur(retouched_resized, kernel_size=ksize, sigma=sigma)

    # High-frequency layer from original (original minus its low-freq)
    high_orig = original_crop - low_orig

    # Combine: ABPN low-freq + original high-freq
    blended = low_reto + high_orig
    return blended.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Padding / tiling utilities (kept for fallback)
# ---------------------------------------------------------------------------

def pad_to_multiple(img, multiple=16):
    """Pad [C, H, W] tensor so H and W are divisible by multiple."""
    _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, [0, pad_w, 0, pad_h], mode='reflect')
    return img, h, w


def tiled_inference(model, image, skin_mask_tensor, tile_size=1024, overlap=128, device='cuda'):
    """Cosine-blended tiling inference. Fallback for large images / no face.

    Args:
        model:            ABPN model
        image:            [3, H, W] float tensor in [0, 1]
        skin_mask_tensor: [1, H, W] float tensor (may be None → ones mask)
        tile_size:        tile dimension
        overlap:          overlap pixels between tiles
        device:           computation device
    Returns:
        output: [3, H, W] retouched tensor
        mask:   [1, H, W] blend tensor from a center tile
    """
    _, H, W = image.shape
    stride = tile_size - overlap

    # Cosine blend window
    ramp = torch.linspace(0, 1, overlap, device=device)
    ones_seg = torch.ones(tile_size - 2 * overlap, device=device)
    window_1d = torch.cat([ramp, ones_seg, ramp.flip(0)])
    window = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)  # [tile, tile]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, tile, tile]

    output = torch.zeros(1, 3, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = image[:, y_start:y_end, x_start:x_end].unsqueeze(0).to(device)
            th, tw = tile.shape[-2:]

            # Build tile mask
            if skin_mask_tensor is not None:
                tile_mask = skin_mask_tensor[:, y_start:y_end, x_start:x_end].unsqueeze(0).to(device)
                if th < tile_size or tw < tile_size:
                    tile_mask = F.pad(tile_mask, [0, tile_size - tw, 0, tile_size - th],
                                      mode='reflect')
            else:
                tile_mask = torch.ones(1, 1, tile_size, tile_size, device=device)

            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, [0, tile_size - tw, 0, tile_size - th], mode='reflect')

            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                out_tile, _ = model(tile, tile_mask)

            out_tile = out_tile.float()[:, :, :th, :tw]
            w = window[:, :, :th, :tw]

            output[:, :, y_start:y_end, x_start:x_end] += out_tile * w
            weight[:, :, y_start:y_end, x_start:x_end] += w

    output = output / weight.clamp(min=1e-8)

    # Mask from a center tile
    center_size = min(tile_size, H, W)
    cy = H // 2 - center_size // 2
    cx = W // 2 - center_size // 2
    center = image[:, cy:cy + center_size, cx:cx + center_size].unsqueeze(0).to(device)
    center_padded, ch, cw = pad_to_multiple(center.squeeze(0), 16)

    if skin_mask_tensor is not None:
        c_mask = skin_mask_tensor[:, cy:cy + center_size, cx:cx + center_size].unsqueeze(0).to(device)
        c_mask, _, _ = pad_to_multiple(c_mask.squeeze(0), 16)
        c_mask = c_mask.unsqueeze(0)
    else:
        c_mask = torch.ones(1, 1, center_padded.shape[-2], center_padded.shape[-1], device=device)

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        _, mask = model(center_padded.unsqueeze(0), c_mask)

    return output.squeeze(0).clamp(0, 1), mask.float().squeeze(0)


# ---------------------------------------------------------------------------
# Main inference function (programmatic API)
# ---------------------------------------------------------------------------

def retouch_image(model, img_tensor, device, tile_size=1024, use_freq_sep=True):
    """Retouch a portrait image using the ABPN crop-and-paste pipeline.

    Args:
        model:       ABPN model (eval mode, on device)
        img_tensor:  [3, H, W] float tensor in [0, 1]
        device:      torch.device
        tile_size:   target square size for ABPN input
        use_freq_sep: apply frequency-separation paste-back (Strategy C)
    Returns:
        retouched:   [3, H, W] float tensor in [0, 1]
        blend_mask:  [1, H, W] float tensor (blend confidence from ABPN)
    """
    _, H, W = img_tensor.shape
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    # Convert to numpy for MediaPipe (uint8 RGB)
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # --- 1. Detect face ---
    box = detect_face_box(img_np)

    if box is None:
        # Fallback: tile the whole image
        print("No face detected — using tiled inference on full image.")
        skin_mask_np = generate_skin_mask(img_np)
        skin_mask_t = torch.from_numpy(skin_mask_np).unsqueeze(0)  # [1, H, W]
        return tiled_inference(model, img_tensor, skin_mask_t,
                               tile_size=tile_size, device=device)

    # --- 2. Crop face region (padded, clamped) ---
    x1, y1, x2, y2 = padded_face_box(box, H, W, padding_frac=0.20)
    crop_h = y2 - y1
    crop_w = x2 - x1

    if crop_h < 32 or crop_w < 32:
        # Crop too small — fall back to full tiling
        print("Face crop too small — using tiled inference on full image.")
        skin_mask_np = generate_skin_mask(img_np)
        skin_mask_t = torch.from_numpy(skin_mask_np).unsqueeze(0)
        return tiled_inference(model, img_tensor, skin_mask_t,
                               tile_size=tile_size, device=device)

    original_crop = img_tensor[:, y1:y2, x1:x2]  # [3, crop_h, crop_w]

    # --- 3. Resize crop to tile_size (square, padded) ---
    resized_crop, pad_info = resize_to_square(original_crop, tile_size)  # [3, tile, tile]

    # --- 4. Generate skin mask for the resized crop ---
    resized_np = (resized_crop.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    skin_mask_np = generate_skin_mask(resized_np)
    skin_mask_t = torch.from_numpy(skin_mask_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    # --- 5. Run ABPN ---
    inp = resized_crop.unsqueeze(0).to(device)  # [1, 3, tile, tile]
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out_resized, blend = model(inp, skin_mask_t)

    out_resized = out_resized.float().squeeze(0).clamp(0.0, 1.0)   # [3, tile, tile]
    blend = blend.float().squeeze(0)                                # [1, tile, tile]

    # --- 6. Upsample ABPN output back to original crop resolution ---
    out_at_crop_res = unpad_and_resize(out_resized, pad_info, crop_h, crop_w)  # [3, crop_h, crop_w]

    # --- 7. Frequency-separation paste-back ---
    if use_freq_sep:
        out_at_crop_res = freq_sep_pasteback(original_crop, out_at_crop_res, sigma=2.0)

    # --- 8. Paste retouched crop into full-resolution image ---
    retouched = img_tensor.clone()
    retouched[:, y1:y2, x1:x2] = out_at_crop_res

    # Build full-res blend mask (zeros outside face crop)
    blend_full = torch.zeros(1, H, W)
    blend_crop = unpad_and_resize(blend, pad_info, crop_h, crop_w)  # [1, crop_h, crop_w]
    blend_full[:, y1:y2, x1:x2] = blend_crop.cpu()

    return retouched.clamp(0.0, 1.0), blend_full


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(retouched, blend_mask, basename, output_dir, mask_only=False):
    """Save retouched PNG and mask PNG to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if not mask_only:
        out_path = os.path.join(output_dir, f'{basename}_retouched.png')
        to_pil_image(retouched.cpu()).save(out_path)
        print(f"Saved: {out_path}")

    mask_path = os.path.join(output_dir, f'{basename}_mask.png')
    # blend_mask may be [1, H, W] or [3, H, W] — take mean over channel dim
    mask_img = blend_mask.mean(dim=0).cpu()  # [H, W]
    mask_img = (mask_img.clamp(0, 1) * 255).byte()
    Image.fromarray(mask_img.numpy(), mode='L').save(mask_path)
    print(f"Saved: {mask_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ABPN face retouching — crop-and-paste pipeline for 20+ MP portraits'
    )
    parser.add_argument('portrait', type=str, help='Input portrait image path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained .pth checkpoint')
    parser.add_argument('-o', '--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('-s', '--strength', type=float, default=1.0,
                        help='Blend strength 0.0–1.0 (1 = full model output)')
    parser.add_argument('--tile', type=int, default=1024,
                        help='Tile / target ABPN input size')
    parser.add_argument('--no-freq-sep', action='store_true',
                        help='Disable frequency-separation paste-back')
    parser.add_argument('--mask-only', action='store_true',
                        help='Save mask PNG only, skip retouched output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)
    print(f"Loaded ABPN from {args.checkpoint} (device={device})")

    # Load image
    img = Image.open(args.portrait).convert('RGB')
    img_tensor = to_tensor(img)  # [3, H, W] in [0, 1]
    _, H, W = img_tensor.shape
    print(f"Input: {args.portrait}  ({W}x{H})")

    basename = os.path.splitext(os.path.basename(args.portrait))[0]

    use_freq_sep = not args.no_freq_sep

    # Run inference
    retouched, blend_mask = retouch_image(
        model, img_tensor, device,
        tile_size=args.tile,
        use_freq_sep=use_freq_sep,
    )

    # Strength blending
    if args.strength < 1.0:
        retouched = img_tensor.to(device) * (1.0 - args.strength) + retouched.to(device) * args.strength
        retouched = retouched.clamp(0.0, 1.0).cpu()

    save_outputs(retouched, blend_mask, basename, args.output, mask_only=args.mask_only)


if __name__ == '__main__':
    main()
