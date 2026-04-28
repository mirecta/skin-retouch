import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (pix2pix style)."""

    def __init__(self, in_ch=6, nf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult_prev = 1
        for i in range(1, n_layers):
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(nf * mult_prev, nf * mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(nf * mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            mult_prev = mult

        # stride-1 layer before output
        mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(nf * mult_prev, nf * mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(nf * mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # final logit layer — no norm, no activation
        layers.append(nn.Conv2d(nf * mult, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
