"""Synthetic checks: skin texture is removed, strong edges survive."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skinretouch import MaskParams, RetouchParams, process  # noqa: E402
from skinretouch.core import laplacian_pyramid, reconstruct  # noqa: E402


def synthetic(h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    skin = np.array([0.85, 0.65, 0.55], np.float32)       # RGB skin tone
    img = np.ones((h, w, 3), np.float32) * skin
    fine = rng.normal(0, 0.03, (h, w)).astype(np.float32)  # pores / noise
    img += fine[..., None]
    for _ in range(25):                                    # blemishes
        x, y = rng.integers(20, w - 20), rng.integers(20, h - 20)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 7)), (0.75, 0.5, 0.45), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    img[200:260, 300:420] = (0.15, 0.1, 0.08)              # dark "eye" = strong edge
    return np.clip(img, 0, 1)


def test_pyramid_roundtrip():
    y = np.random.default_rng(1).random((300, 257)).astype(np.float32)
    assert np.abs(reconstruct(laplacian_pyramid(y, 4)) - y).max() < 1e-5


def test_texture_removed_edges_kept():
    img = synthetic()
    out, mask = process(img, RetouchParams(fine=80, medium=60, large=20, threshold=30),
                        MaskParams())
    flat = (slice(20, 180), slice(20, 280))
    std_in = img[flat].std(axis=(0, 1)).mean()
    std_out = out[flat].std(axis=(0, 1)).mean()
    edge_in = np.ptp(img[230, 290:310], axis=0).mean()
    edge_out = np.ptp(out[230, 290:310], axis=0).mean()
    print(f"flat std {std_in:.4f} -> {std_out:.4f}; edge contrast {edge_in:.3f} -> {edge_out:.3f}; "
          f"mask mean on skin {mask[flat].mean():.2f}, on eye {mask[215:245, 330:390].mean():.2f}")
    assert std_out < std_in * 0.6, "skin texture not reduced enough"
    assert edge_out > edge_in * 0.9, "edge was blurred"
    assert mask[flat].mean() > 0.8 and mask[215:245, 330:390].mean() < 0.2


def test_healing_removes_blemishes():
    img = synthetic()
    base, _ = process(img, RetouchParams(threshold=30), MaskParams())
    healed, _ = process(img, RetouchParams(threshold=30, healing=70), MaskParams())
    flat = (slice(20, 180), slice(20, 280))
    s0 = base[flat].std(axis=(0, 1)).mean()
    s1 = healed[flat].std(axis=(0, 1)).mean()
    edge = np.ptp(healed[230, 290:310], axis=0).mean()
    print(f"healing: flat std {s0:.4f} -> {s1:.4f}, eye edge {edge:.3f}")
    assert s1 < s0 * 0.8, "blemishes not healed"
    assert edge > 0.5, "eye edge damaged"


if __name__ == "__main__":
    test_healing_removes_blemishes()
    test_pyramid_roundtrip()
    test_texture_removed_edges_kept()
    img = synthetic()
    out, _ = process(img, RetouchParams(fine=80, medium=60, large=20, threshold=30), MaskParams())
    p = os.path.join(os.path.dirname(__file__), "synthetic_compare.png")
    cv2.imwrite(p, cv2.cvtColor((np.hstack([img, out]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("ok, wrote", p)
