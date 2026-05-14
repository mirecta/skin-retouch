"""
StyleRetoucher — portrait retouching with StyleGAN2 priors.

Based on: Wang et al., "StyleRetoucher: Generalized Portrait Image Retouching
with GAN Priors", arxiv 2312.14389 (Dec 2023).

Architecture:
  SemanticExtractor (SE):  input image → W+ latent + multi-res feature pyramid
  BAFS:                    per-resolution attention blending of SE features
                           with frozen StyleGAN2 intermediate features
  StyleGAN2 prior:         frozen pretrained FFHQ generator
  Output:                  retouched 1024×1024 face on StyleGAN2 manifold

The crucial property: the output is constrained to the StyleGAN2 prior
manifold, so it cannot regress to a mushy mean — every output looks like
a real, sharp face.
"""

import sys, os, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# Channels at each StyleGAN2 resolution stage (FFHQ-1024 config-f).
# Output channels of block b{res} = in_channels of block b{2*res}.
SG2_CHANNELS = {
    4: 512, 8: 512, 16: 512, 32: 512, 64: 512,
    128: 256, 256: 128, 512: 64, 1024: 32,
}


# ---------------------------------------------------------------------------
# Semantic Extractor — encoder that maps input image to W+ + feature pyramid
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SemanticExtractor(nn.Module):
    """
    Encodes input image (1024×1024) into:
      - W+ latent: [B, 18, 512]
      - Feature pyramid at each StyleGAN2 resolution

    Architecture mirrors StyleGAN2 generator structure inverted.
    """

    def __init__(self):
        super().__init__()
        # 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4
        # ch:     32 → 64  → 128 → 256 → 512 → 512 → 512 → 512 → 512
        self.from_rgb   = nn.Conv2d(3, 32, 1)
        self.down_1024  = _conv_bn_relu(32,  32, stride=1)        # 1024×1024
        self.down_512   = _conv_bn_relu(32,  64, stride=2)        # 512×512
        self.down_256   = _conv_bn_relu(64,  128, stride=2)       # 256×256
        self.down_128   = _conv_bn_relu(128, 256, stride=2)       # 128×128
        self.down_64    = _conv_bn_relu(256, 512, stride=2)       # 64×64
        self.down_32    = _conv_bn_relu(512, 512, stride=2)       # 32×32
        self.down_16    = _conv_bn_relu(512, 512, stride=2)       # 16×16
        self.down_8     = _conv_bn_relu(512, 512, stride=2)       # 8×8
        self.down_4     = _conv_bn_relu(512, 512, stride=2)       # 4×4

        # W+ extraction: aggregate 4×4 features into 18×512 latent
        self.w_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 18 * 512),
        )

    def forward(self, x):
        """
        x: [B, 3, 1024, 1024] in [0, 1]
        Returns:
          ws:       [B, 18, 512]
          features: dict {resolution: tensor [B, C_res, res, res]}
        """
        # Normalise to [-1, 1] to match StyleGAN2 input convention
        x = x * 2.0 - 1.0
        h = self.from_rgb(x)

        feats = {}
        h = self.down_1024(h);  feats[1024] = h
        h = self.down_512(h);   feats[512]  = h
        h = self.down_256(h);   feats[256]  = h
        h = self.down_128(h);   feats[128]  = h
        h = self.down_64(h);    feats[64]   = h
        h = self.down_32(h);    feats[32]   = h
        h = self.down_16(h);    feats[16]   = h
        h = self.down_8(h);     feats[8]    = h
        h = self.down_4(h);     feats[4]    = h

        ws = self.w_head(h).view(-1, 18, 512)
        return ws, feats


# ---------------------------------------------------------------------------
# Blemish-Aware Feature Selection (BAFS)
# ---------------------------------------------------------------------------

class BAFS(nn.Module):
    """
    Blends SE features with StyleGAN2 features using spatial + channel attention.

    out = (M_s · M_c) * f_SE + (1 - M_s · M_c) * f_SG

    M_s: spatial mask [B, 1, H, W] — where SE features matter (e.g., non-blemish)
    M_c: channel mask [B, C, 1, 1] — which channels favor SE over SG
    """

    def __init__(self, channels):
        super().__init__()
        # Spatial attention: looks at concatenated features at full spatial res
        self.spatial = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4 if channels >= 4 else channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 4 if channels >= 4 else channels, 1, 3, padding=1),
        )
        # Channel attention: looks at global-pooled features
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, f_se, f_sg):
        """
        f_se: SE feature  [B, C, H, W]
        f_sg: SG feature  [B, C, H, W]
        Returns: blended  [B, C, H, W]
        """
        cat = torch.cat([f_se, f_sg], dim=1)
        m_s = torch.sigmoid(self.spatial(cat))
        m_c = torch.sigmoid(self.channel(cat))
        mask = m_s * m_c
        return mask * f_se + (1.0 - mask) * f_sg


# ---------------------------------------------------------------------------
# StyleRetoucher — full pipeline
# ---------------------------------------------------------------------------

class StyleRetoucher(nn.Module):
    """
    Forward pass:
      1. SE(input) → ws, se_features
      2. StyleGAN2 mapping(ws) is bypassed — we use ws from SE directly
      3. Run StyleGAN2 synthesis block-by-block; after each block, BAFS-blend
         the block's output features with the corresponding SE features
      4. Final output is the StyleGAN2 RGB at 1024×1024
    """

    # Stages at which we apply BAFS blending (skip 4×4 since SG2 starts there)
    BLEND_RESOLUTIONS = [8, 16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, stylegan2_pkl):
        super().__init__()
        repo_path = os.path.join(os.path.dirname(__file__), 'stylegan3_repo')
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        with open(stylegan2_pkl, 'rb') as f:
            G = pickle.load(f)['G_ema']

        # Freeze StyleGAN2 — only SE + BAFS train
        for p in G.parameters():
            p.requires_grad = False
        self.sg2 = G

        self.se = SemanticExtractor()

        # Initialise SE's w_head to start near StyleGAN2's average W+. We use
        # small-magnitude weights + bias=w_avg so the first forward outputs
        # roughly w_avg, and gradients can move it toward input-specific W+.
        with torch.no_grad():
            z = torch.randn(10000, G.z_dim)
            ws = G.mapping(z, None, truncation_psi=1.0)
            w_avg = ws.mean(dim=0).flatten()              # [18*512]
            linear = self.se.w_head[-1]
            linear.weight.data.mul_(0.01)                 # tiny but non-zero
            linear.bias.data.copy_(w_avg)

        self.bafs = nn.ModuleDict({
            str(res): BAFS(SG2_CHANNELS[res])
            for res in self.BLEND_RESOLUTIONS
        })

    @property
    def trainable_parameters(self):
        return list(self.se.parameters()) + list(self.bafs.parameters())

    def forward(self, x):
        """
        x: [B, 3, 1024, 1024] in [0, 1]
        Returns: [B, 3, 1024, 1024] retouched output in [0, 1]
        """
        B = x.shape[0]
        ws, se_feats = self.se(x)

        # Manual synthesis loop with BAFS blending.
        # StyleGAN2 ws sharing: each block reads (num_conv + num_torgb) codes,
        # but the next block starts only num_conv later — torgb overlaps with
        # the next block's first conv. See SynthesisNetwork.forward for details.
        block_ws = []
        ws_idx = 0
        for res in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            block = getattr(self.sg2.synthesis, f'b{res}')
            n_ws = block.num_conv + block.num_torgb
            block_ws.append(ws.narrow(1, ws_idx, n_ws))
            ws_idx += block.num_conv

        x_feat = None
        img = None
        for res, w_block in zip([4, 8, 16, 32, 64, 128, 256, 512, 1024], block_ws):
            block = getattr(self.sg2.synthesis, f'b{res}')
            with torch.amp.autocast('cuda', enabled=False):
                x_feat, img = block(x_feat, img, w_block, noise_mode='const',
                                    force_fp32=True)
            if str(res) in self.bafs:
                f_se = se_feats[res]
                x_feat = self.bafs[str(res)](f_se, x_feat)

        # StyleGAN2 outputs in [-1, 1]; convert to [0, 1]
        out = (img.clamp(-1, 1) + 1.0) / 2.0
        return out
