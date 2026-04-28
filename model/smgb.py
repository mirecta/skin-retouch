"""Soft Mask Generation Branch — U-Net architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 + BN + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool2x2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat  # downsampled, skip


class DecoderBlock(nn.Module):
    """Upsample + concat skip + Conv3x3."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SMGB(nn.Module):
    """
    Soft Mask Generation Branch.

    Input:  [B, 3, H, W]
    Output: [B, 3, H//4, W//4] — soft blemish mask Mpred
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder: 4 blocks
        self.enc1 = EncoderBlock(in_channels, 64)   # -> H/2
        self.enc2 = EncoderBlock(64, 128)            # -> H/4
        self.enc3 = EncoderBlock(128, 256)           # -> H/8
        self.enc4 = EncoderBlock(256, 512)           # -> H/16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        # Decoder: 2 blocks going back to H/4 (not full res)
        # skip4 is at H/8 (pre-pool size of enc4), skip3 is at H/4 (pre-pool of enc3)
        self.dec4 = DecoderBlock(512, 512, 256)   # H/16 -> H/8 (using skip4)
        self.dec3 = DecoderBlock(256, 256, 128)   # H/8  -> H/4 (using skip3)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(128, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder — skip is pre-pool (full conv output)
        x1, skip1 = self.enc1(x)     # skip1: [B,  64, H,   W  ], x1: H/2
        x2, skip2 = self.enc2(x1)    # skip2: [B, 128, H/2, W/2], x2: H/4
        x3, skip3 = self.enc3(x2)    # skip3: [B, 256, H/4, W/4], x3: H/8
        x4, skip4 = self.enc4(x3)    # skip4: [B, 512, H/8, W/8], x4: H/16

        # Bottleneck
        b = self.bottleneck(x4)       # [B, 512, H/16, W/16]

        # Decoder (back to H/4)
        d4 = self.dec4(b, skip4)      # [B, 256, H/8, W/8]
        d3 = self.dec3(d4, skip3)     # [B, 128, H/4, W/4]

        return self.head(d3)          # [B, 3, H/4, W/4]
