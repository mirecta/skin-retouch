"""Finishing 'Enhancements' (applied after smoothing, optionally masked)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhanceParams:
    sharpness: float = 0.0   # 0..100 unsharp mask on luma
    softness: float = 0.0    # 0..100 soft glow
    warmth: float = 0.0      # -100..100
    tint: float = 0.0        # -100..100 (+ = magenta, - = green)
    brightness: float = 0.0  # -100..100
    contrast: float = 0.0    # -100..100


def enhance(rgb: np.ndarray, e: EnhanceParams, mask: np.ndarray | None = None) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    sigma = max(1.0, min(rgb.shape[:2]) / 600.0)

    if e.sharpness > 0:
        ycc = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        y = ycc[..., 0]
        ycc[..., 0] = y + (y - cv2.GaussianBlur(y, (0, 0), sigma)) * (e.sharpness / 100.0) * 1.5
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

    if e.softness > 0:
        glow = cv2.GaussianBlur(out, (0, 0), sigma * 8)
        screen = 1.0 - (1.0 - out) * (1.0 - glow)
        out = out + (screen - out) * (e.softness / 100.0) * 0.5

    if e.warmth or e.tint:
        wa, ti = e.warmth / 100.0 * 0.08, e.tint / 100.0 * 0.05
        out = out + np.array([wa + ti * 0.5, -ti, -wa + ti * 0.5], np.float32)

    if e.brightness:
        out = out + e.brightness / 100.0 * 0.2

    if e.contrast:
        k = 1.0 + e.contrast / 100.0 * 0.5
        out = (out - 0.5) * k + 0.5

    out = np.clip(out, 0.0, 1.0)
    if mask is not None:
        out = rgb + (out - rgb) * mask[..., None]
    return out
