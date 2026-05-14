"""
RetouchLoss for ABPN (Lei et al., CVPR 2022).
  MSE(R_0, target) + MSE(R_l, downsample(target))   lambda_mse=1.0
  Perceptual/LPIPS(R_0, target)                      lambda_perc=0.1
  LSGAN adversarial                                  lambda_adv=0.1
  Dice(M, mask_gt)  — diff-based pseudo-GT           lambda_dice=1.0
  TV on blend maps                                   lambda_tv=0.1
  Aesthetic reward  relu(quality(inp)-quality(out))  lambda_aesthetic=0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Perceptual (LPIPS-style VGG)
# ---------------------------------------------------------------------------

class _PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg[:9]))    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg[9:18]))  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, output, target):
        out_n = (output - self.mean) / self.std
        tgt_n = (target - self.mean) / self.std
        f1_o = self.slice1(out_n);  f1_t = self.slice1(tgt_n)
        f2_o = self.slice2(f1_o);   f2_t = self.slice2(f1_t)
        return F.l1_loss(f1_o, f1_t) + F.l1_loss(f2_o, f2_t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tv_loss(x):
    return (torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
          + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean())


def _dice_loss(pred, gt):
    p = pred.flatten(1).float()
    g = gt.flatten(1).float()
    inter = (p * g).sum(1)
    return (1.0 - (2.0 * inter + 1e-6) / (p.sum(1) + g.sum(1) + 1e-6)).mean()


def _downsample2x(x):
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
# Aesthetic reward (CLIP-IQA via pyiqa)
# Penalises only when output scores worse than the raw input:
#   loss = mean(relu(quality(inp) - quality(out)))
# This is one-sided — never pushes the model to be overly aggressive,
# just ensures retouching doesn't degrade perceived quality.
# ---------------------------------------------------------------------------

class _AestheticRewardLoss(nn.Module):
    def __init__(self, metric: str = 'clipiqa'):
        super().__init__()
        try:
            import pyiqa
            self.scorer = pyiqa.create_metric(metric, as_loss=True)
            self.scorer.eval()
            for p in self.scorer.parameters():
                p.requires_grad = False
            self.enabled = True
        except ImportError:
            self.enabled = False
            print('[AestheticRewardLoss] pyiqa not installed — loss disabled. '
                  'Run: uv pip install pyiqa')

    def forward(self, output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=output.device)
        # pyiqa expects [0,1] float32, size ≥ 32×32
        score_out = self.scorer(output.float())
        score_inp = self.scorer(inp.float())
        return torch.relu(score_inp - score_out).mean()


# ---------------------------------------------------------------------------
# Combined RetouchLoss
# ---------------------------------------------------------------------------

class RetouchLoss(nn.Module):
    """
    Args (forward):
      R_0:           [B,3,H,W]      full-res generator output
      R_l:           [B,3,H/4,W/4] low-res LRL output
      M:             [B,1,H/4,W/4] predicted mask
      blends:        list of blend maps [B_l, B_1, B_0]
      target:        [B,3,H,W]      ground-truth retouched
      inp:           [B,3,H,W]      original input (for aesthetic reward)
      D_fake_logits: patch logits or None
      mask_gt:       [B,1,H,W] or None — diff-based pseudo GT mask
    """

    def __init__(self, lambda_mse=1.0, lambda_perc=0.1, lambda_adv=0.1,
                 lambda_dice=1.0, lambda_tv=0.1, lambda_aesthetic=0.0,
                 aesthetic_metric='clipiqa'):
        super().__init__()
        self.lambda_mse       = lambda_mse
        self.lambda_perc      = lambda_perc
        self.lambda_adv       = lambda_adv
        self.lambda_dice      = lambda_dice
        self.lambda_tv        = lambda_tv
        self.lambda_aesthetic = lambda_aesthetic

        self.perc_fn = _PerceptualLoss()
        if lambda_aesthetic > 0.0:
            self.aesthetic_fn = _AestheticRewardLoss(metric=aesthetic_metric)
        else:
            self.aesthetic_fn = None

    def forward(self, R_0, R_l, M, blends, target, inp=None,
                D_fake_logits=None, mask_gt=None):

        # MSE at full res + low res
        target_l = _downsample2x(_downsample2x(target))
        mse_full = F.mse_loss(R_0, target)
        mse_low  = F.mse_loss(R_l, target_l)
        mse      = mse_full + mse_low

        # Perceptual
        perc = self.perc_fn(R_0.float(), target.float())

        # Adversarial (LSGAN generator loss)
        adv = torch.tensor(0.0, device=R_0.device)
        if D_fake_logits is not None:
            adv = 0.5 * ((D_fake_logits - 1.0) ** 2).mean()

        # Dice on predicted mask
        dice = torch.tensor(0.0, device=R_0.device)
        if mask_gt is not None and self.lambda_dice > 0:
            mgt = F.interpolate(mask_gt, size=M.shape[-2:], mode='bilinear', align_corners=False)
            dice = _dice_loss(M, mgt)

        # TV on blend maps
        tv = sum(_tv_loss(b) for b in blends) / len(blends)

        # Aesthetic reward: penalise only if output looks worse than input
        aesthetic = torch.tensor(0.0, device=R_0.device)
        if self.aesthetic_fn is not None and inp is not None:
            aesthetic = self.aesthetic_fn(R_0.float(), inp.float())

        total = (self.lambda_mse       * mse
               + self.lambda_perc      * perc
               + self.lambda_adv       * adv
               + self.lambda_dice      * dice
               + self.lambda_tv        * tv
               + self.lambda_aesthetic * aesthetic)

        loss_dict = {
            'mse':       mse.item(),
            'perc':      perc.item(),
            'adv':       adv.item(),
            'dice':      dice.item(),
            'tv':        tv.item(),
            'aesthetic': aesthetic.item(),
            'total':     total.item(),
        }
        return total, loss_dict
