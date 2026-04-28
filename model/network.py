"""FaceRetouchNet — top-level model assembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .smgb import SMGB
from .fsr import FSR
from .mrf import MRF


def laplacian_pyramid(x, levels=2):
    """Build Laplacian pyramid via average pooling.

    Returns:
        pyramid: [I0 (full), I1 (1/2), I2 (1/4)]
    """
    pyramid = [x]
    for _ in range(levels):
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        pyramid.append(x)
    return pyramid


def high_freq_components(pyramid):
    """Compute high-frequency residuals between pyramid levels.

    Returns:
        hf: [H0, H1] where H_i = I_i - upsample(I_{i+1})
    """
    hf = []
    for i in range(len(pyramid) - 1):
        up = F.interpolate(pyramid[i + 1], size=pyramid[i].shape[-2:],
                           mode='bilinear', align_corners=False)
        hf.append(pyramid[i] - up)
    return hf


class FaceRetouchNet(nn.Module):
    """
    Full face retouching network.

    Forward:
        1. Build Laplacian pyramid [I0, I1, I2]
        2. Compute HF components [H0, H1]
        3. SMGB on I2 -> Mpred [B, 3, H/4, W/4]
        4. FSR at each scale with interpolated mask
        5. MRF fusion -> output [B, 3, H, W]

    Args:
        channels: base feature channels (default 64)
        fsr_blocks: number of FSR blocks per scale (default 4)
        window_size: SDP window size (default 8)
        patch_size: SFFN patch size (default 8)
        levels: Laplacian pyramid depth (default 2)
    """

    def __init__(self, channels=64, fsr_blocks=4, window_size=8,
                 patch_size=8, levels=2):
        super().__init__()
        self.levels = levels

        self.smgb = SMGB(in_channels=3)
        self.fsr = FSR(channels=channels, num_blocks=fsr_blocks,
                       window_size=window_size, patch_size=patch_size)
        self.mrf = MRF()

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input portrait

        Returns:
            output: [B, 3, H, W] retouched image
            Mpred:  [B, 3, H/4, W/4] predicted soft blemish mask
        """
        # 1. Laplacian pyramid
        pyramid = laplacian_pyramid(x, self.levels)  # [I0, I1, I2]
        hf = high_freq_components(pyramid)             # [H0, H1]

        # 2. Soft mask prediction (full-res input, outputs at H/4)
        Mpred = self.smgb(x)

        # 3. FSR at each scale
        R = []
        for i, Ii in enumerate(pyramid):
            # Interpolate mask to this scale
            Mp_i = F.interpolate(Mpred, size=Ii.shape[-2:],
                                 mode='bilinear', align_corners=False)
            R.append(self.fsr(Ii, Mp_i))

        # 4. MRF fusion
        output = self.mrf(R, hf)

        return output, Mpred

    @classmethod
    def from_config(cls, cfg):
        """Create model from config dict."""
        return cls(
            channels=cfg.get('channels', 64),
            fsr_blocks=cfg.get('fsr_blocks', 4),
            window_size=cfg.get('window_size', 8),
            patch_size=cfg.get('patch_size', 8),
            levels=cfg.get('levels', 2),
        )
