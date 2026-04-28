"""Frequency Selection & Restoration module: FADA -> SDP -> SFFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class GEGLU(nn.Module):
    """GEGLU activation: x[:, :C//2] * sigmoid(x[:, C//2:])."""

    def forward(self, x):
        c = x.shape[1] // 2
        return x[:, :c] * torch.sigmoid(x[:, c:])


class FADA(nn.Module):
    """
    Frequency-Aware Dynamic Aggregation.

    Applies learnable quantization to magnitude and phase in frequency domain.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Pre-projection to match channels
        self.pre_conv = nn.Conv2d(channels, channels, 1)

        # Aggregation via 1x1 conv + GEGLU
        self.agg = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            GEGLU(),
        )

        # Learnable quantization matrices — initialized to ones (neutral)
        # These are spatial-frequency domain weights, registered as buffers
        # and resized dynamically based on input size
        self.W_A = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.W_phi = nn.Parameter(torch.ones(1, channels, 1, 1))

        # Learnable residual gate — initialized so signal passes through
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, mask=None):
        """
        Args:
            z:    [B, C, H, W] spatial features
            mask: [B, 3, H, W] or None — soft blemish mask
        """
        if mask is not None:
            # Interpolate mask to match z spatial size if needed
            if mask.shape[-2:] != z.shape[-2:]:
                mask = F.interpolate(mask, size=z.shape[-2:], mode='bilinear',
                                     align_corners=False)
            # Expand mask channels to match z
            if mask.shape[1] != z.shape[1]:
                mask = mask.mean(dim=1, keepdim=True).expand_as(z)
            z_masked = z * mask
        else:
            z_masked = z

        z_masked = self.pre_conv(z_masked)

        # FFT requires float32
        orig_dtype = z_masked.dtype
        z_masked = z_masked.float()

        # Channel-wise 2D DFT
        F_z = torch.fft.fft2(z_masked, norm='ortho')
        A = F_z.abs()          # magnitude
        phi = F_z.angle()      # phase

        # Aggregate magnitude (keep scale bounded)
        A_agg = self.agg(A)

        # Learnable quantization as scaling factors (sigmoid keeps in [0, 2])
        scale_A = 2.0 * torch.sigmoid(self.W_A)
        scale_phi = 2.0 * torch.sigmoid(self.W_phi) - 1.0  # [-1, 1]

        A_prime = scale_A * A_agg
        phi_prime = phi + scale_phi * 0.5

        # Recompose
        F_prime = torch.polar(A_prime, phi_prime)

        # Learnable residual gate (sigmoid → [0,1])
        alpha = torch.sigmoid(self.gate)
        F_out = alpha * F_prime + (1 - alpha) * F_z

        # IDFT back to spatial
        z_tilde = torch.fft.ifft2(F_out, norm='ortho').real
        z_tilde = z_tilde.clamp(-10.0, 10.0).to(orig_dtype)

        return z_tilde


class SDP(nn.Module):
    """
    Space Domain Projection — window-based self-attention.

    Uses FADA for Q/K/V projections and Swin-style windowed attention.
    """

    def __init__(self, channels, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size

        self.norm = nn.LayerNorm(channels)

        # Q, K, V projections via FADA
        self.fada_q = FADA(channels)
        self.fada_k = FADA(channels)
        self.fada_v = FADA(channels)

        self.scale = channels ** -0.5

        # Learnable relative position bias [ws*ws, ws*ws]
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(window_size * window_size, window_size * window_size)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z):
        """z: [B, C, H, W]"""
        B, C, H, W = z.shape
        ws = self.window_size

        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            z_pad = F.pad(z, [0, pad_w, 0, pad_h], mode='reflect')
        else:
            z_pad = z

        _, _, Hp, Wp = z_pad.shape

        # LayerNorm (channel-last)
        z_norm = rearrange(z_pad, 'b c h w -> b h w c')
        z_norm = self.norm(z_norm)
        z_norm = rearrange(z_norm, 'b h w c -> b c h w')

        # Q, K, V via FADA
        Q = self.fada_q(z_norm)
        K = self.fada_k(z_norm)
        V = self.fada_v(z_norm)

        # Window partition: [B, C, H, W] -> [B*nW, C, ws, ws]
        nH, nW_count = Hp // ws, Wp // ws
        Q = rearrange(Q, 'b c (nh ws1) (nw ws2) -> (b nh nw) c ws1 ws2',
                       ws1=ws, ws2=ws)
        K = rearrange(K, 'b c (nh ws1) (nw ws2) -> (b nh nw) c ws1 ws2',
                       ws1=ws, ws2=ws)
        V = rearrange(V, 'b c (nh ws1) (nw ws2) -> (b nh nw) c ws1 ws2',
                       ws1=ws, ws2=ws)

        # Flatten spatial dims for attention: [BnW, C, ws*ws]
        Q = rearrange(Q, 'b c h w -> b c (h w)')
        K = rearrange(K, 'b c h w -> b c (h w)')
        V = rearrange(V, 'b c h w -> b c (h w)')

        # Attention: [BnW, ws*ws, ws*ws]
        attn = torch.bmm(Q.transpose(1, 2), K) * self.scale
        attn = attn + self.rel_pos_bias
        attn = F.softmax(attn, dim=-1)

        # Apply to V
        out = torch.bmm(attn, V.transpose(1, 2))  # [BnW, ws*ws, C]
        out = rearrange(out, 'b (h w) c -> b c h w', h=ws, w=ws)

        # Window reverse
        out = rearrange(out, '(b nh nw) c ws1 ws2 -> b c (nh ws1) (nw ws2)',
                        nh=nH, nw=nW_count)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        out = self.proj_out(out)
        return out + z  # residual


class SFFN(nn.Module):
    """
    Selective Frequency Feed-Forward Network.

    JPEG-inspired learnable frequency quantization in patch space.
    """

    def __init__(self, channels, patch_size=8):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        self.norm = nn.LayerNorm(channels)
        self.pre_conv = nn.Conv2d(channels, channels, 1)

        # Learnable quantization matrix
        self.W = nn.Parameter(torch.ones(1, channels, patch_size, patch_size))

        # GEGLU output
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            GEGLU(),
        )

    def forward(self, z):
        """z: [B, C, H, W]"""
        B, C, H, W = z.shape
        ps = self.patch_size

        # LayerNorm
        z_norm = rearrange(z, 'b c h w -> b h w c')
        z_norm = self.norm(z_norm)
        z_norm = rearrange(z_norm, 'b h w c -> b c h w')

        z1 = self.pre_conv(z_norm)

        # Pad to patch_size multiples
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            z1 = F.pad(z1, [0, pad_w, 0, pad_h], mode='reflect')

        _, _, Hp, Wp = z1.shape

        # Patch unfold: [B, C, nH, nW, ps, ps]
        z1 = rearrange(z1, 'b c (nh p1) (nw p2) -> (b nh nw) c p1 p2',
                        p1=ps, p2=ps)

        # DFT per patch (requires float32)
        orig_dtype = z1.dtype
        z1 = z1.float()
        z1_fft = torch.fft.fft2(z1, norm='ortho')

        # Learnable quantization (sigmoid-bounded to prevent blowup)
        scale_W = 2.0 * torch.sigmoid(self.W)
        z1_fft = scale_W * z1_fft

        # IDFT
        z1 = torch.fft.ifft2(z1_fft, norm='ortho').real
        z1 = z1.clamp(-10.0, 10.0).to(orig_dtype)

        # Patch fold
        nH, nW_count = Hp // ps, Wp // ps
        z1 = rearrange(z1, '(b nh nw) c p1 p2 -> b c (nh p1) (nw p2)',
                        nh=nH, nw=nW_count)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            z1 = z1[:, :, :H, :W]

        # GEGLU + residual
        z_out = self.post_conv(z1)
        return z_out + z


class FSRBlock(nn.Module):
    """Single FSR block: FADA -> SDP -> SFFN."""

    def __init__(self, channels, window_size=8, patch_size=8):
        super().__init__()
        self.fada = FADA(channels)
        self.sdp = SDP(channels, window_size)
        self.sffn = SFFN(channels, patch_size)

    def forward(self, z, mask=None):
        z = self.fada(z, mask)
        z = self.sdp(z)
        z = self.sffn(z)
        return z


class FSR(nn.Module):
    """
    Frequency Selection & Restoration module.

    Applies num_blocks FSR blocks sequentially.
    Input image is first projected to feature channels, then back to 3ch.
    """

    def __init__(self, channels=64, num_blocks=4, window_size=8, patch_size=8):
        super().__init__()
        self.in_proj = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            FSRBlock(channels, window_size, patch_size)
            for _ in range(num_blocks)
        ])
        self.out_proj = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, 3, H, W] — image at some pyramid scale
            mask: [B, 3, H, W] — soft blemish mask (interpolated to this scale)
        Returns:
            [B, 3, H, W] — restored image at same scale
        """
        z = self.in_proj(x)
        for block in self.blocks:
            z = checkpoint(block, z, mask, use_reentrant=False)
        return self.out_proj(z) + x  # global residual
