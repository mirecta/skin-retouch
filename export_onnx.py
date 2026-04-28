"""Export trained FaceRetouchNet to ONNX format for C++/Rust inference."""

import argparse
import os
import torch
import torch.nn as nn
import yaml

from model.network import FaceRetouchNet


class FaceRetouchNetExport(nn.Module):
    """Wrapper that returns only the retouched image (no mask tuple)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, mask = self.model(x)
        return output, mask


class FaceRetouchNetImageOnly(nn.Module):
    """Wrapper that returns only the retouched image."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _ = self.model(x)
        return output


def main():
    parser = argparse.ArgumentParser(description='Export FaceRetouchNet to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained .pth checkpoint')
    parser.add_argument('--output', type=str, default='face_retouch.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size the ONNX model expects (default 512)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic spatial axes (any H/W multiple of 16)')
    parser.add_argument('--image-only', action='store_true',
                        help='Export model with single output (image only, no mask)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default 17)')
    parser.add_argument('--simplify', action='store_true',
                        help='Run onnx-simplifier after export')
    args = parser.parse_args()

    device = torch.device('cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']

    # Build model
    model = FaceRetouchNet.from_config(cfg['model'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    if args.image_only:
        export_model = FaceRetouchNetImageOnly(model)
        output_names = ['retouched']
    else:
        export_model = FaceRetouchNetExport(model)
        output_names = ['retouched', 'mask']

    export_model.eval()

    # Dummy input
    tile = args.tile_size
    dummy = torch.randn(1, 3, tile, tile)

    # Dynamic axes
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'retouched': {0: 'batch', 2: 'height', 3: 'width'},
        }
        if not args.image_only:
            dynamic_axes['mask'] = {0: 'batch', 2: 'height', 3: 'width'}
    else:
        dynamic_axes = None

    # Export
    print(f"Exporting to {args.output} (tile={tile}, opset={args.opset}, "
          f"dynamic={args.dynamic})")

    torch.onnx.export(
        export_model,
        dummy,
        args.output,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({size_mb:.1f} MB)")

    # Verify
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    # Optional simplify
    if args.simplify:
        try:
            import onnxsim
            onnx_model, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(onnx_model, args.output)
                size_mb = os.path.getsize(args.output) / (1024 * 1024)
                print(f"Simplified: {args.output} ({size_mb:.1f} MB)")
            else:
                print("Warning: simplification failed, keeping original")
        except ImportError:
            print("Install onnx-simplifier: pip install onnxsim")

    # Test with ONNX Runtime if available
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(args.output)
        inp = np.random.randn(1, 3, tile, tile).astype(np.float32)
        outputs = sess.run(None, {'input': inp})

        print(f"\nONNX Runtime test:")
        print(f"  Input:     [1, 3, {tile}, {tile}]")
        print(f"  Retouched: {outputs[0].shape}")
        if len(outputs) > 1:
            print(f"  Mask:      {outputs[1].shape}")
        print("  OK!")
    except ImportError:
        print("\nInstall onnxruntime to verify: pip install onnxruntime")


if __name__ == '__main__':
    main()
