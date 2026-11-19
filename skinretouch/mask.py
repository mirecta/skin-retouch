"""Skin masks.

Two sources, combined by multiplication or max:
* colour mask: soft distance from a skin-tone centre in the CrCb plane
  (Portraiture's Hue / Saturation / Latitude / Fuzziness / Feathering sliders);
* optional face detection (OpenCV YuNet) that samples the skin tone from the
  face itself and restricts work to the people in the photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Typical skin centre in YCrCb (0..1 scale) across many skin tones.
DEFAULT_CR, DEFAULT_CB = 150 / 255.0, 110 / 255.0


@dataclass
class MaskParams:
    latitude: float = 50.0    # 0..100 how wide the accepted colour range is
    fuzziness: float = 50.0   # 0..100 softness of the colour range edge
    feather: float = 30.0     # 0..100 spatial blur of the mask
    min_luma: float = 0.12    # ignore very dark pixels (hair, shadows)
    max_luma: float = 0.98    # ignore blown highlights


def sample_skin_tone(ycc: np.ndarray, region: np.ndarray | None = None) -> tuple[float, float]:
    """Robust (median) skin tone from a region mask, or the default centre."""
    if region is None or region.sum() < 50:
        return DEFAULT_CR, DEFAULT_CB
    cr = ycc[..., 1][region > 0]
    cb = ycc[..., 2][region > 0]
    return float(np.median(cr)), float(np.median(cb))


def color_mask(rgb: np.ndarray, mp: MaskParams, centre: tuple[float, float] | None = None) -> np.ndarray:
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    cr0, cb0 = centre if centre else (DEFAULT_CR, DEFAULT_CB)
    d = np.sqrt((ycc[..., 1] - cr0) ** 2 + (ycc[..., 2] - cb0) ** 2)
    radius = 0.02 + 0.10 * (mp.latitude / 100.0)
    soft = 0.005 + 0.08 * (mp.fuzziness / 100.0)
    m = np.clip((radius + soft - d) / (2 * soft), 0.0, 1.0)
    y = ycc[..., 0]
    m *= np.clip((y - mp.min_luma) / 0.05, 0, 1) * np.clip((mp.max_luma - y) / 0.02, 0, 1)
    if mp.feather > 0:
        sigma = max(0.5, min(rgb.shape[:2]) * 0.01 * mp.feather / 100.0)
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return m.astype(np.float32)


class FaceDetector:
    """Thin wrapper over cv2.FaceDetectorYN (needs the YuNet ONNX model file).

    Download: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    (face_detection_yunet_2023mar.onnx, ~230 kB).
    """

    def __init__(self, model_path: str, score: float = 0.7):
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score, 0.3, 5000)

    def detect(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        scale = min(1.0, 1024 / max(h, w))
        small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None:
            return []
        return [tuple(int(v / scale) for v in f[:4]) for f in faces]


def face_region(shape: tuple[int, ...], faces, grow: float = 1.6) -> np.ndarray:
    """Soft elliptical region around detected faces (covers neck/ears too)."""
    m = np.zeros(shape[:2], np.float32)
    for x, y, w, h in faces:
        c = (int(x + w / 2), int(y + h / 2))
        axes = (int(w * grow / 2), int(h * grow / 2 * 1.15))
        cv2.ellipse(m, c, axes, 0, 0, 360, 1.0, -1)
    if m.any():
        m = cv2.GaussianBlur(m, (0, 0), max(2.0, min(shape[:2]) * 0.02))
    return m


def core_sample_region(shape, faces) -> np.ndarray:
    """Central cheek/forehead area of each face, used to sample skin tone."""
    m = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in faces:
        cv2.rectangle(m, (x + w // 4, y + h // 4), (x + 3 * w // 4, y + 3 * h // 4), 1, -1)
    return m


class FaceParser:
    """BiSeNet face parsing (CelebAMask-HQ, 19 classes) via cv2.dnn.

    Model: https://github.com/yakhyo/face-parsing (MIT), resnet18.onnx, 53 MB.
    Trained on face crops, so it is run on an expanded crop around each face.
    """

    CLASSES = ["background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
               "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip",
               "neck", "neck_l", "cloth", "hair", "hat"]
    SKIN = (1, 7, 8, 10, 14)  # skin, ears, nose, neck
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: str, expand: float = 2.0):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.expand = expand

    def crop_box(self, shape, face) -> tuple[int, int, int, int]:
        x, y, w, h = face
        side = int(max(w, h) * self.expand)
        cx, cy = x + w // 2, y + h // 2 - int(h * 0.05)
        return cx - side // 2, cy - side // 2, side, side

    def probs(self, rgb: np.ndarray, box) -> np.ndarray:
        """Class probabilities (19, side, side) for a (possibly out-of-image) box."""
        x, y, s, _ = box
        H, W = rgb.shape[:2]
        crop = np.zeros((s, s, 3), np.float32)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + s, W), min(y + s, H)
        crop[y0 - y:y1 - y, x0 - x:x1 - x] = rgb[y0:y1, x0:x1]
        inp = (cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA) - self.MEAN) / self.STD
        self.net.setInput(inp.transpose(2, 0, 1)[None].astype(np.float32))
        logits = self.net.forward()[0]
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    def skin_mask(self, rgb: np.ndarray, faces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (skin probability, coverage, nose probability), full size.

        coverage = 1 inside parsed crops (feathered at their border), 0 elsewhere.
        The nose map lets Skin Healing skip nostrils and nose shading.
        """
        H, W = rgb.shape[:2]
        skin = np.zeros((H, W), np.float32)
        cover = np.zeros((H, W), np.float32)
        nose = np.zeros((H, W), np.float32)
        for face in faces:
            box = self.crop_box(rgb.shape, face)
            x, y, s, _ = box
            p = self.probs(rgb, box)
            no = cv2.resize(p[10], (s, s), interpolation=cv2.INTER_LINEAR)
            sk = p[list(self.SKIN)].sum(axis=0)
            sk = np.clip((sk - 0.2) / 0.5, 0, 1)  # confident >= 0.7, none <= 0.2
            sk = cv2.resize(sk, (s, s), interpolation=cv2.INTER_LINEAR)
            # feather the crop border so it blends with the colour mask outside
            ramp = np.minimum(np.arange(s), np.arange(s)[::-1]) / (s * 0.08)
            border = np.clip(np.minimum.outer(ramp, ramp), 0, 1).astype(np.float32)
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + s, W), min(y + s, H)
            sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
            skin[y0:y1, x0:x1] = np.maximum(skin[y0:y1, x0:x1], sk[sl] * border[sl])
            cover[y0:y1, x0:x1] = np.maximum(cover[y0:y1, x0:x1], border[sl])
            nose[y0:y1, x0:x1] = np.maximum(nose[y0:y1, x0:x1], no[sl])
        return skin, cover, nose


def build_mask(rgb: np.ndarray, mp: MaskParams, detector: FaceDetector | None = None,
               faces_only: bool = False, parser: FaceParser | None = None
               ) -> tuple[np.ndarray, list, np.ndarray]:
    """Returns (retouch mask, faces, healing mask)."""
    faces = detector.detect(rgb) if detector else []
    centre = None
    if faces:
        ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        centre = sample_skin_tone(ycc, core_sample_region(rgb.shape, faces))
    m = color_mask(rgb, mp, centre)
    if faces and faces_only:
        m *= face_region(rgb.shape, faces)
    if faces and parser is not None:
        skin, cover, nose = parser.skin_mask(rgb, faces)
        # edge-aware refinement: snaps the coarse network mask to image edges
        if hasattr(cv2, "ximgproc"):
            r = max(4, int(np.median([max(f[2], f[3]) for f in faces]) / 80))
            skin = np.clip(cv2.ximgproc.guidedFilter(rgb.astype(np.float32), skin, r, 1e-4), 0, 1)
        if mp.feather > 0:
            sigma = max(0.5, min(rgb.shape[:2]) * 0.004 * mp.feather / 100.0)
            skin = cv2.GaussianBlur(skin, (0, 0), sigma)
        # inside parsed face crops trust the network, outside keep the colour mask
        m = skin * cover + m * (1.0 - cover)
        heal = m * (1.0 - np.clip(cv2.dilate(nose, None, iterations=3) * 2, 0, 1))
        return m.astype(np.float32), faces, heal.astype(np.float32)
    return m.astype(np.float32), faces, m.astype(np.float32)
