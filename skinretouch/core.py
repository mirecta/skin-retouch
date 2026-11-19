"""Core skin-smoothing algorithm (Portraiture-style, clean-room).

Pipeline
--------
1. Convert to YCrCb (float32, 0..1). Texture lives mostly in luma (Y).
2. Build a Laplacian pyramid of Y. Each band holds details of one size:
   band 0-1 = fine (pores, noise), band 2 = medium (blemishes),
   band 3+ = large (uneven skin, shadows).
3. Attenuate each band with an amplitude-dependent gain:
       gain(d) = 1 - strength * exp(-(d / T)^2)
   Small-amplitude detail (|d| << T, skin texture) is removed, large-amplitude
   detail (|d| >> T, real edges: eyes, lips, hair) passes through. This is the
   same idea as a bilateral filter, applied per frequency band.
4. Reconstruct, optionally smooth the chroma a little (colour uniformity),
   then blend with the original through the skin mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RetouchParams:
    fine: float = 60.0         # 0..100  smoothing of finest detail
    medium: float = 40.0       # 0..100
    large: float = 15.0        # 0..100
    threshold: float = 20.0    # 0..100  amplitude regarded as "skin texture"
    portrait_size: float = 0.0  # pixels of face height; 0 = auto from image size
    uniformity: float = 20.0   # 0..100  chroma smoothing
    edge_protect: float = 50.0  # 0..100  extra protection of strong edges
    amount: float = 100.0      # 0..100  final blend with original
    healing: float = 0.0       # 0..100  remove small spots (blemishes, specks)


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def num_levels(shape: tuple[int, ...], portrait_size: float) -> int:
    """Pick pyramid depth so the coarsest band matches large skin structures.

    Portraiture exposes this as 'Portrait Size'; bigger faces need more levels.
    """
    ref = portrait_size if portrait_size > 0 else min(shape[:2]) * 0.6
    # coarsest band ~ ref / 16 pixels wide
    levels = int(np.clip(np.round(np.log2(max(ref, 32) / 16.0)), 2, 6))
    return levels


def laplacian_pyramid(y: np.ndarray, levels: int) -> list[np.ndarray]:
    bands = []
    cur = y
    for _ in range(levels):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        bands.append(cur - up)
        cur = down
    bands.append(cur)  # residual low-pass
    return bands


def reconstruct(bands: list[np.ndarray]) -> np.ndarray:
    cur = bands[-1]
    for band in reversed(bands[:-1]):
        cur = cv2.pyrUp(cur, dstsize=(band.shape[1], band.shape[0])) + band
    return cur


def band_strengths(levels: int, p: RetouchParams) -> list[float]:
    """Map Fine/Medium/Large sliders onto the pyramid bands."""
    f, m, l = p.fine / 100.0, p.medium / 100.0, p.large / 100.0
    out = []
    for i in range(levels):
        t = i / max(levels - 1, 1)  # 0 = finest, 1 = coarsest
        if t <= 0.5:
            s = f + (m - f) * (t / 0.5)
        else:
            s = m + (l - m) * ((t - 0.5) / 0.5)
        out.append(float(np.clip(s, 0.0, 1.0)))
    return out


def edge_map(y: np.ndarray, sigma: float, lo: float = 0.15, hi: float = 0.30) -> np.ndarray:
    """0..1 map of strong, coherent edges (eyes, lips, hairline, jaw).

    The gradient is converted to an estimated *step height* in luma units, so
    the map is absolute: small blemishes (low contrast) are not protected,
    real features with contrast above `hi` are fully protected.
    """
    blur = cv2.GaussianBlur(y, (0, 0), sigma)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    # Sobel 3x3 = 8x derivative; a blurred step of height h peaks at h/(2.5*sigma)
    step = np.sqrt(gx * gx + gy * gy) / 8.0 * 2.5 * sigma
    e = np.clip((step - lo) / (hi - lo), 0.0, 1.0)
    return cv2.GaussianBlur(e, (0, 0), sigma)


def smooth_luma(y: np.ndarray, p: RetouchParams) -> np.ndarray:
    levels = num_levels(y.shape, p.portrait_size)
    bands = laplacian_pyramid(y, levels)
    strengths = band_strengths(levels, p)
    # threshold slider 0..100 -> amplitude 0.005 .. 0.20 of full range
    base_t = 0.005 + (p.threshold / 100.0) * 0.195
    for i, s in enumerate(strengths):
        if s <= 0:
            continue
        d = bands[i]
        # coarser bands carry larger amplitudes for the same visual contrast
        t = base_t * (1.0 + 0.35 * i)
        gain = 1.0 - s * np.exp(-(d / t) ** 2)
        bands[i] = d * gain
    out = reconstruct(bands)
    if p.edge_protect > 0:
        e = edge_map(y, sigma=max(1.5, min(y.shape[:2]) / 300.0))
        w = e * (p.edge_protect / 100.0)
        out = out * (1.0 - w) + y * w
    return out


def find_spots(y: np.ndarray, k: int, t: float, inner: np.ndarray | None,
               max_spots: int = 2000) -> list[tuple[int, int, float]]:
    """Scale-normalized LoG blob detection. Returns (x, y, radius) of spots.

    A spot is a local extremum of s^2*LoG at its own scale, round (Hessian
    eigenvalues of similar size) and with contrast above t.
    """
    spots = []
    sigmas = np.geomspace(max(1.0, k / 10.0), k / 4.5, 4)
    for s in sigmas:
        yb = cv2.GaussianBlur(y, (0, 0), s)
        ixx = cv2.Sobel(yb, cv2.CV_32F, 2, 0, ksize=3, scale=0.25)
        iyy = cv2.Sobel(yb, cv2.CV_32F, 0, 2, ksize=3, scale=0.25)
        ixy = cv2.Sobel(yb, cv2.CV_32F, 1, 1, ksize=3, scale=0.25)
        log = (ixx + iyy) * s * s                      # >0 dark spot, <0 bright speck
        det = ixx * iyy - ixy * ixy
        round_ = det / ((ixx + iyy) ** 2 / 4.0 + 1e-12)
        r = max(3, int(s * 2) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        for sign in (1.0, -1.0):
            L = log * sign
            peak = (L >= cv2.dilate(L, ker)) & (L > t) & (round_ > 0.5)
            if inner is not None:
                peak &= inner > 0.5
            ys, xs = np.nonzero(peak)
            for yy, xx in zip(ys, xs):
                spots.append((float(L[yy, xx]), int(xx), int(yy), float(s * 1.414 * 1.4)))
    spots.sort(reverse=True)
    return [(x, y_, r) for _, x, y_, r in spots[:max_spots]]


def heal_spots(y: np.ndarray, p: RetouchParams, mask: np.ndarray | None = None):
    """Skin Healing: replace small round spots (blemishes) with surrounding skin.

    Spots are detected individually (find_spots) and only a disc of the spot's
    own size is replaced by 'clean' skin (grey closing+opening, which removes
    dark/bright structures smaller than k). Lines (lip edge, nostril, brow) are
    not round, features outside the inner skin mask are never touched.
    """
    if p.healing <= 0:
        return y, None, 0
    ref_size = p.portrait_size if p.portrait_size > 0 else min(y.shape[:2]) * 0.6
    k = int(np.clip(ref_size / 25.0, 5, 51)) | 1        # max spot diameter
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(cv2.morphologyEx(y, cv2.MORPH_CLOSE, se), cv2.MORPH_OPEN, se)
    clean = cv2.GaussianBlur(clean, (0, 0), k / 6.0)

    inner = None
    if mask is not None:
        # close first: a blemish is redder than skin and punches a hole in the mask
        # (only holes up to spot size k; eyes and mouth are bigger and stay excluded)
        filled = cv2.morphologyEx(mask.astype(np.float32), cv2.MORPH_CLOSE, se)
        inner = cv2.erode(np.clip((filled - 0.6) / 0.3, 0, 1), se)

    t = 0.006 + 0.03 * (1.0 - p.healing / 100.0)        # minimal spot strength
    w = np.zeros_like(y)
    amp = np.abs(y - clean)
    for x, yy, r in find_spots(y, k, t, inner):
        if amp[yy, x] > 0.22:   # very dark/contrasty = nostril, eye corner, not a blemish
            continue
        cv2.circle(w, (x, yy), int(round(r)), 1.0, -1, lineType=cv2.LINE_AA)
    w = cv2.GaussianBlur(w, (0, 0), max(1.0, k / 12.0))
    w = np.clip(w * 1.3, 0, 1) * min(1.0, p.healing / 50.0)
    if inner is not None:
        w *= np.clip(inner * 2, 0, 1)
    return y + (clean - y) * w, w, k


def heal_chroma(c: np.ndarray, w: np.ndarray | None, k: int) -> np.ndarray:
    """Fill healed spots with the average chroma of the skin around them.

    Normalized convolution: spots (weight w) are left out of the average.
    """
    if w is None:
        return c
    keep = 1.0 - w
    sigma = max(1.0, k / 2.0)
    ref = cv2.GaussianBlur(c * keep, (0, 0), sigma) / (cv2.GaussianBlur(keep, (0, 0), sigma) + 1e-6)
    return c + (ref - c) * w


def smooth_chroma(cc: np.ndarray, guide: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return cc
    r = max(2, int(min(guide.shape[:2]) * 0.01))
    eps = 1e-4
    if hasattr(cv2, "ximgproc"):
        sm = cv2.ximgproc.guidedFilter(guide, cc, r, eps)
    else:
        sm = cv2.bilateralFilter(cc, 2 * r + 1, 0.05, r)
    return cc + (sm - cc) * strength


def retouch(img: np.ndarray, p: RetouchParams, mask: np.ndarray | None = None,
            heal_mask: np.ndarray | None = None) -> np.ndarray:
    """Retouch an RGB image (uint8/uint16/float). Returns float32 RGB 0..1.

    mask: optional float 0..1 (H, W); where retouching is applied.
    """
    rgb = to_float(img[..., :3])
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0].copy()
    yh, hw, hk = heal_spots(y, p, heal_mask if heal_mask is not None else mask)
    y2 = smooth_luma(yh, p)
    u = p.uniformity / 100.0
    cr = smooth_chroma(heal_chroma(ycc[..., 1].copy(), hw, hk), yh, u)
    cb = smooth_chroma(heal_chroma(ycc[..., 2].copy(), hw, hk), yh, u)
    out = cv2.cvtColor(np.dstack([y2, cr, cb]).astype(np.float32), cv2.COLOR_YCrCb2RGB)
    w = p.amount / 100.0
    if mask is not None:
        w = mask[..., None].astype(np.float32) * w
    out = rgb + (out - rgb) * w
    return np.clip(out, 0.0, 1.0)
