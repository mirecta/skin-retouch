"""
ABPN: Adaptive Blend Pyramid Network (Lei et al., CVPR 2022).

Architecture:
  LRL (Local Retouching Layer):
    MutualEncoder → 6-block CBR feature pyramid on I_l (= I_0 / 4)
    MPB (Mask Prediction Branch) → M at I_l scale
    LRB (Local Retouching Branch) → R_l at I_l scale via 3 LAM decoder layers
  BPL (Blend Pyramid Layer):
    R-ABM: invert R_l → B_l
    RefiningModule × 2: B_l → B_1 → B_0 using H_1, H_0 (Laplacian high-freq)
    ABM: I_0, B_0 → R_0 (full-res output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LAM(nn.Module):
    """Local Attentive Module: out = tanh(W_f(cat(skip, up(feat), M))) * sigmoid(W_g(...))"""

    def __init__(self, skip_ch, feat_ch, out_ch):
        super().__init__()
        in_ch = skip_ch + feat_ch + 1
        self.conv_f = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv_g = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, skip, feat, M):
        feat_up = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        M_up   = F.interpolate(M,    size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, feat_up, M_up], dim=1)
        return torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))


class RefiningModule(nn.Module):
    """B_out = φ_2(h(φ_1(cat(up(B), H)))) + up(B)  — progressive blend refinement."""

    def __init__(self):
        super().__init__()
        self.phi1 = nn.Conv2d(6, 16, 3, padding=1)
        self.phi2 = nn.Conv2d(16, 3,  3, padding=1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, B, H):
        B_up = F.interpolate(B, size=H.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([B_up, H], dim=1)
        return self.phi2(self.act(self.phi1(x))) + B_up


class ABM(nn.Module):
    """
    Adaptive Blend Module: R = Σ_i (j_i·B + k_i) · g(I, i)
    g(I,0)=1, g(I,1)=I, g(I,2)=I²
    Init: j=[0,1,0], k=[0,0,0] → R = B·I (identity in log-space)
    """

    def __init__(self):
        super().__init__()
        self.j = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))
        self.k = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def _poly(self, I):
        g0 = torch.ones_like(I)
        g1 = I
        g2 = I * I
        sum_jg = self.j[0] * g0 + self.j[1] * g1 + self.j[2] * g2
        sum_kg = self.k[0] * g0 + self.k[1] * g1 + self.k[2] * g2
        return sum_jg, sum_kg

    def forward(self, I, B):
        sum_jg, sum_kg = self._poly(I)
        return sum_jg * B + sum_kg

    def reverse(self, I, R):
        sum_jg, sum_kg = self._poly(I)
        return (R - sum_kg) / (sum_jg.abs() + 1e-6)


class MutualEncoder(nn.Module):
    """6-block CBR encoder on I_l; returns (f1, f2, f3, f4, f5, f6)."""

    def __init__(self):
        super().__init__()
        self.b1 = _cbr(3,   64,  stride=1)   # f1: same as I_l (H/4 of I_0)
        self.b2 = _cbr(64,  128, stride=2)   # f2: H/8
        self.b3 = _cbr(128, 256, stride=2)   # f3: H/16
        self.b4 = _cbr(256, 256, stride=2)   # f4: H/32
        self.b5 = _cbr(256, 256, stride=1)   # f5: H/32
        self.b6 = _cbr(256, 256, stride=1)   # f6: H/32

    def forward(self, Il):
        f1 = self.b1(Il)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        f5 = self.b5(f4)
        f6 = self.b6(f5)
        return f1, f2, f3, f4, f5, f6


class MPB(nn.Module):
    """Mask Prediction Branch: f3 (H/16) → M at I_l scale (H/4) via 4 CBR + upsample."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _cbr(256, 256, stride=1),
            _cbr(256, 128, stride=1),
            _cbr(128, 64,  stride=1),
            _cbr(64,  32,  stride=1),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, f3):
        m = self.net(f3)
        m = F.interpolate(m, scale_factor=4, mode='bilinear', align_corners=False)
        return torch.sigmoid(m)


class LRB(nn.Module):
    """Local Retouching Branch: 3 LAM decoder layers → R_l at I_l scale."""

    def __init__(self):
        super().__init__()
        self.lam3 = LAM(skip_ch=256, feat_ch=256, out_ch=256)  # (f3, f6) → d3 at H/16
        self.lam2 = LAM(skip_ch=128, feat_ch=256, out_ch=128)  # (f2, d3) → d2 at H/8
        self.lam1 = LAM(skip_ch=64,  feat_ch=128, out_ch=64)   # (f1, d2) → d1 at H/4
        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, f1, f2, f3, f6, M):
        d3 = self.lam3(f3, f6, M)
        d2 = self.lam2(f2, d3, M)
        d1 = self.lam1(f1, d2, M)
        return torch.sigmoid(self.out_conv(d1))


class ABPN(nn.Module):
    """
    Adaptive Blend Pyramid Network.

    forward(x) -> (R_0, R_l, M, blends)
      x:       [B, 3, H, W] in [0, 1]
      R_0:     [B, 3, H, W]     final full-res retouched output
      R_l:     [B, 3, H/4, W/4] low-res retouched (MSE supervision)
      M:       [B, 1, H/4, W/4] predicted skin/retouching mask
      blends:  [B_l, B_1, B_0]  blend maps (TV regularisation)
    """

    def __init__(self):
        super().__init__()
        self.encoder  = MutualEncoder()
        self.mpb      = MPB()
        self.lrb      = LRB()
        self.abm      = ABM()
        self.refine1  = RefiningModule()   # B_l → B_1 via H_1
        self.refine2  = RefiningModule()   # B_1 → B_0 via H_0

    @staticmethod
    def _ds2(x):
        h, w = x.shape[-2:]
        return F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=False)

    @staticmethod
    def _us(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def _laplacian(self, I0):
        I1 = self._ds2(I0)
        Il = self._ds2(I1)
        H1 = I1 - self._us(Il, I1.shape[-2:])
        H0 = I0 - self._us(I1, I0.shape[-2:])
        return Il, H0, H1

    def forward(self, x):
        Il, H0, H1 = self._laplacian(x)

        f1, f2, f3, f4, f5, f6 = self.encoder(Il)
        M   = self.mpb(f3)
        R_l = self.lrb(f1, f2, f3, f6, M)

        B_l = self.abm.reverse(Il, R_l)
        B_1 = self.refine1(B_l, H1)
        B_0 = self.refine2(B_1, H0)
        R_0 = self.abm.forward(x, B_0).clamp(0.0, 1.0)

        return R_0, R_l, M, [B_l, B_1, B_0]

    @classmethod
    def from_config(cls, cfg):
        return cls()
