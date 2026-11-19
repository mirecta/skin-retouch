"""Clean-room Portraiture-style skin retouching."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import RetouchParams, retouch, to_float
from .enhance import EnhanceParams, enhance
from .mask import FaceDetector, FaceParser, MaskParams, build_mask

__all__ = ["RetouchParams", "MaskParams", "EnhanceParams", "FaceDetector", "FaceParser",
           "process", "retouch", "enhance", "build_mask"]


def process(img: np.ndarray, rp: RetouchParams, mp: MaskParams | None = None,
            ep: EnhanceParams | None = None, detector: FaceDetector | None = None,
            use_mask: bool = True, faces_only: bool = False,
            parser: FaceParser | None = None):
    """Full pipeline. Returns (float32 RGB 0..1, mask or None)."""
    rgb = to_float(img[..., :3])
    mask, faces, heal = None, [], None
    if use_mask:
        mask, faces, heal = build_mask(rgb, mp or MaskParams(), detector, faces_only, parser)
    elif detector is not None:
        faces = detector.detect(rgb)
    if faces and rp.portrait_size <= 0:
        # Portraiture's 'Portrait Size': scale band sizes to the actual face
        rp = replace(rp, portrait_size=float(np.median([max(f[2], f[3]) for f in faces])))
    out = retouch(rgb, rp, mask, heal)
    if ep is not None:
        out = enhance(out, ep, mask)
    return out, mask
