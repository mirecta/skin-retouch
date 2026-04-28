"""Loss functions: L1 + Perceptual (VGG) + Mask BCE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-16 perceptual loss at relu2_2 and relu3_3 (per paper)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # relu2_2 = features[:9], relu3_3 = features[9:18]
        self.slice1 = nn.Sequential(*list(features[:9]))
        self.slice2 = nn.Sequential(*list(features[9:18]))

        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        # relu2_2 features
        pred_f1 = self.slice1(pred)
        target_f1 = self.slice1(target)
        loss1 = F.l1_loss(pred_f1, target_f1)

        # relu3_3 features
        pred_f2 = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)
        loss2 = F.l1_loss(pred_f2, target_f2)

        return loss1 + loss2


class RetouchLoss(nn.Module):
    """
    Paper loss (Xu et al. ICCV 2025):
        total = L1(output, target)
              + lambda_mask       * BCE(Mpred, Mgt, label_smoothing=0.05)
              + lambda_perceptual * VGG(output, target)  [relu2_2 + relu3_3]
    """

    def __init__(self, lambda_mask=0.1, lambda_perceptual=0.01, lambda_residual=0.0):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_perceptual = lambda_perceptual

        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.perceptual = PerceptualLoss()

    def forward(self, output, Mpred, target, Mgt, source=None):
        """
        Args:
            output: [B, 3, H, W] model output
            Mpred:  [B, 3, H/4, W/4] predicted mask (sigmoid output)
            target: [B, 3, H, W] ground truth retouched
            Mgt:    [B, 3, H, W] ground truth blemish mask
            source: unused, kept for API compatibility
        """
        # L1 reconstruction loss
        l_retouch = self.l1(output, target)

        # Mask loss with label smoothing 0.05 (per paper) and numerical clamp.
        # Smoothing: scale soft targets to [0.025, 0.975] to avoid overconfidence.
        Mgt_down = F.interpolate(Mgt, size=Mpred.shape[-2:],
                                 mode='bilinear', align_corners=False)
        with torch.autocast(device_type='cuda', enabled=False):
            Mpred_safe = Mpred.float().clamp(1e-6, 1.0 - 1e-6)
            Mgt_smooth = Mgt_down.float() * 0.95 + 0.025
            l_mask = self.bce(Mpred_safe, Mgt_smooth)

        # Perceptual loss (relu2_2 + relu3_3)
        l_perceptual = self.perceptual(output, target)

        total = (l_retouch
                 + self.lambda_mask * l_mask
                 + self.lambda_perceptual * l_perceptual)

        return total, {
            'l_retouch': l_retouch.item(),
            'l_mask': l_mask.item(),
            'l_perceptual': l_perceptual.item(),
            'l_residual': 0.0,
            'total': total.item(),
        }
