"""
ABPN: Adaptive Blend Pyramid Network (Lei et al., CVPR 2022).

nf=64  → paper-original size   (~6.6M params)
nf=128 → 2× wider, higher quality (~25M params, recommended for non-realtime)

Architecture:
  LRL:  MutualEncoder (6-block CBR on I_l=I_0/4)
        MPB (mask prediction from f3)
        LRB (3× LAM decoder → R_l)
  BPL:  R-ABM (invert R_l → B_l)
        RefiningModule × 2 (B_l → B_1 → B_0 via Laplacian high-freq)
        ABM (I_0 + B_0 → R_0)
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
    """Local Attentive Module: out = tanh(W_f(cat(skip,up(feat),M))) * sigmoid(W_g(...))"""

    def __init__(self, skip_ch, feat_ch, out_ch):
        super().__init__()
        in_ch = skip_ch + feat_ch + 1
        self.conv_f = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv_g = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, skip, feat, M):
        feat_up = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        M_up    = F.interpolate(M,    size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, feat_up, M_up], dim=1)
        return torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))


class RefiningModule(nn.Module):
    """B_out = φ_2(h(φ_1(cat(up(B), H)))) + up(B)"""

    def __init__(self):
        super().__init__()
        self.phi1 = nn.Conv2d(6, 16, 3, padding=1)
        self.phi2 = nn.Conv2d(16, 3,  3, padding=1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, B, H):
        B_up = F.interpolate(B, size=H.shape[-2:], mode='bilinear', align_corners=False)
        return self.phi2(self.act(self.phi1(torch.cat([B_up, H], dim=1)))) + B_up


class ABM(nn.Module):
    """R = Σ_i (j_i·B + k_i)·g(I,i); g=(1, I, I²); init j=[0,1,0], k=[0,0,0]"""

    def __init__(self):
        super().__init__()
        self.j = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))
        self.k = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def _poly(self, I):
        g0, g1, g2 = torch.ones_like(I), I, I * I
        return (self.j[0]*g0 + self.j[1]*g1 + self.j[2]*g2,
                self.k[0]*g0 + self.k[1]*g1 + self.k[2]*g2)

    def forward(self, I, B):
        jg, kg = self._poly(I)
        return jg * B + kg

    def reverse(self, I, R):
        jg, kg = self._poly(I)
        return (R - kg) / (jg.abs() + 1e-6)


class MutualEncoder(nn.Module):
    """6-block CBR feature pyramid. nf=64 → [64,128,256,256,256,256] (paper-original)"""

    def __init__(self, nf=64):
        super().__init__()
        c = [nf, nf*2, nf*4, nf*4, nf*4, nf*4]
        self.b1 = _cbr(3,    c[0], stride=1)
        self.b2 = _cbr(c[0], c[1], stride=2)
        self.b3 = _cbr(c[1], c[2], stride=2)
        self.b4 = _cbr(c[2], c[3], stride=2)
        self.b5 = _cbr(c[3], c[4], stride=1)
        self.b6 = _cbr(c[4], c[5], stride=1)
        self.ch = c

    def forward(self, Il):
        f1 = self.b1(Il)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        f5 = self.b5(f4)
        f6 = self.b6(f5)
        return f1, f2, f3, f4, f5, f6


class MPB(nn.Module):
    """Mask Prediction Branch: f3 → M at I_l scale."""

    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            _cbr(in_ch,      in_ch,      stride=1),
            _cbr(in_ch,      in_ch // 2, stride=1),
            _cbr(in_ch // 2, in_ch // 4, stride=1),
            _cbr(in_ch // 4, in_ch // 8, stride=1),
            nn.Conv2d(in_ch // 8, 1, 1),
        )

    def forward(self, f3):
        m = self.net(f3)
        m = F.interpolate(m, scale_factor=4, mode='bilinear', align_corners=False)
        return torch.sigmoid(m)


class LRB(nn.Module):
    """Local Retouching Branch: 3 LAM decoder → R_l."""

    def __init__(self, nf=64):
        super().__init__()
        c1, c2, c4 = nf, nf*2, nf*4
        self.lam3    = LAM(skip_ch=c4, feat_ch=c4, out_ch=c4)
        self.lam2    = LAM(skip_ch=c2, feat_ch=c4, out_ch=c2)
        self.lam1    = LAM(skip_ch=c1, feat_ch=c2, out_ch=c1)
        self.out_conv = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, f1, f2, f3, f6, M):
        d3 = self.lam3(f3, f6, M)
        d2 = self.lam2(f2, d3, M)
        d1 = self.lam1(f1, d2, M)
        return torch.sigmoid(self.out_conv(d1))


class ABPN(nn.Module):
    """
    Adaptive Blend Pyramid Network.
    forward(x) → (R_0, R_l, M, [B_l, B_1, B_0])
    """

    def __init__(self, nf=64):
        super().__init__()
        self.encoder = MutualEncoder(nf)
        self.mpb     = MPB(in_ch=nf * 4)
        self.lrb     = LRB(nf)
        self.abm     = ABM()
        self.refine1 = RefiningModule()
        self.refine2 = RefiningModule()

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
        return cls(nf=cfg.get('nf', 64))
