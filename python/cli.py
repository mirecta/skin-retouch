#!/usr/bin/env python3
"""Command-line skin retouch: python cli.py in.jpg [-o out.jpg] [--compare]."""

import argparse
import os

import cv2
import numpy as np

from skinretouch import EnhanceParams, FaceDetector, FaceParser, MaskParams, RetouchParams, process


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    a = ap.add_argument
    a("--fine", type=float, default=60); a("--medium", type=float, default=40)
    a("--large", type=float, default=15); a("--threshold", type=float, default=20)
    a("--portrait-size", type=float, default=0); a("--uniformity", type=float, default=20)
    a("--edge-protect", type=float, default=50); a("--amount", type=float, default=100)
    a("--latitude", type=float, default=50); a("--fuzziness", type=float, default=50)
    a("--feather", type=float, default=30)
    a("--healing", type=float, default=0, help="0..100 remove small spots")
    a("--no-mask", action="store_true", help="retouch whole image")
    a("--face-model", help="YuNet ONNX model for face detection")
    a("--parse-model", help="BiSeNet face-parsing ONNX (needs --face-model)")
    a("--faces-only", action="store_true", help="limit mask to detected faces")
    a("--sharpness", type=float, default=0); a("--softness", type=float, default=0)
    a("--warmth", type=float, default=0); a("--tint", type=float, default=0)
    a("--brightness", type=float, default=0); a("--contrast", type=float, default=0)
    a("--compare", action="store_true", help="also write before|after image")
    a("--save-mask", action="store_true")
    args = ap.parse_args()

    bgr = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise SystemExit(f"cannot read {args.input}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)

    rp = RetouchParams(args.fine, args.medium, args.large, args.threshold,
                       args.portrait_size, args.uniformity, args.edge_protect, args.amount,
                       args.healing)
    mp = MaskParams(args.latitude, args.fuzziness, args.feather)
    ep = EnhanceParams(args.sharpness, args.softness, args.warmth, args.tint,
                       args.brightness, args.contrast)
    det = FaceDetector(args.face_model) if args.face_model else None
    parser = FaceParser(args.parse_model) if args.parse_model else None

    out, mask = process(rgb, rp, mp, ep, det, not args.no_mask, args.faces_only, parser)

    base, ext = os.path.splitext(args.output or args.input)
    out_path = args.output or f"{base}_retouched{ext}"
    maxv = 65535 if bgr.dtype == np.uint16 else 255
    dtype = np.uint16 if bgr.dtype == np.uint16 else np.uint8
    to_save = lambda x: cv2.cvtColor((x * maxv + 0.5).astype(dtype), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, to_save(out))
    print("wrote", out_path)
    if args.compare:
        src = rgb.astype(np.float32) / maxv
        p = f"{base}_compare.jpg"
        cv2.imwrite(p, cv2.cvtColor((np.hstack([src, out]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print("wrote", p)
    if args.save_mask and mask is not None:
        p = f"{base}_mask.png"
        cv2.imwrite(p, (mask * 255).astype(np.uint8))
        print("wrote", p)


if __name__ == "__main__":
    main()
