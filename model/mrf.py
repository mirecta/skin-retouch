"""Multi-Resolution Fusion module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLC(nn.Module):
    """Conv-LeakyReLU-Conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class MRF(nn.Module):
    """
    Multi-Resolution Fusion.

    Progressive bottom-up fusion of FSR outputs with high-frequency components.

    Inputs:
        R = [R0, R1, R2]  — FSR outputs at scales 1/1, 1/2, 1/4
        H = [H0, H1]      — high-frequency Laplacian components at 1/1, 1/2

    Fusion (bottom-up, l=2 -> l=0):
        O2 = R2
        O1 = CLC(cat(up(O2), H1)) + R1
        O0 = CLC(cat(up(O1), H0)) + R0
    """

    def __init__(self):
        super().__init__()
        # CLC blocks: input is cat(upsampled_prev=3ch, HF=3ch) = 6ch -> 3ch out
        self.clc1 = CLC(6, 3)  # fuse scale 2 -> 1
        self.clc0 = CLC(6, 3)  # fuse scale 1 -> 0

    def forward(self, R, H):
        """
        Args:
            R: list [R0, R1, R2] each [B, 3, H_i, W_i]
            H: list [H0, H1] each [B, 3, H_i, W_i]
        Returns:
            output: [B, 3, H, W] full-resolution retouched image
        """
        R0, R1, R2 = R
        H0, H1 = H

        # Start from coarsest scale
        O2 = R2

        # Fuse scale 2 -> 1
        O2_up = F.interpolate(O2, size=R1.shape[-2:], mode='bilinear',
                              align_corners=False)
        O1 = self.clc1(torch.cat([O2_up, H1], dim=1)) + R1

        # Fuse scale 1 -> 0
        O1_up = F.interpolate(O1, size=R0.shape[-2:], mode='bilinear',
                              align_corners=False)
        O0 = self.clc0(torch.cat([O1_up, H0], dim=1)) + R0

        return O0
