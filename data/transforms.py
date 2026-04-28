import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class PairedTransform:
    """Augmentations applied identically to input/target pairs."""

    def __init__(self, patch_size=512, augment=True):
        self.patch_size = patch_size
        self.augment = augment

    def __call__(self, input_img, target_img):
        """
        Args:
            input_img:  Tensor [3, H, W] range [0, 1]
            target_img: Tensor [3, H, W] range [0, 1]
        Returns:
            input_img, target_img: both [3, patch_size, patch_size]
        """
        _, h, w = input_img.shape

        if self.augment:
            # Random crop
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                input_img = input_img[:, top:top + self.patch_size, left:left + self.patch_size]
                target_img = target_img[:, top:top + self.patch_size, left:left + self.patch_size]
            else:
                input_img = TF.resize(input_img, [self.patch_size, self.patch_size],
                                      antialias=True)
                target_img = TF.resize(target_img, [self.patch_size, self.patch_size],
                                       antialias=True)

            # Random horizontal flip
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)

            # Random 90-degree rotation
            if random.random() < 0.3:
                k = random.choice([1, 2, 3])
                input_img = torch.rot90(input_img, k, [1, 2])
                target_img = torch.rot90(target_img, k, [1, 2])

            # Mild color jitter (applied identically via same seed)
            if random.random() < 0.5:
                brightness = 1.0 + random.uniform(-0.05, 0.05)
                contrast = 1.0 + random.uniform(-0.05, 0.05)
                input_img = TF.adjust_brightness(input_img, brightness)
                input_img = TF.adjust_contrast(input_img, contrast)
                target_img = TF.adjust_brightness(target_img, brightness)
                target_img = TF.adjust_contrast(target_img, contrast)
                input_img = input_img.clamp(0, 1)
                target_img = target_img.clamp(0, 1)
        else:
            # Validation/test: center crop or resize to 1024
            if h >= 1024 and w >= 1024:
                top = (h - 1024) // 2
                left = (w - 1024) // 2
                input_img = input_img[:, top:top + 1024, left:left + 1024]
                target_img = target_img[:, top:top + 1024, left:left + 1024]
            else:
                input_img = TF.resize(input_img, [1024, 1024], antialias=True)
                target_img = TF.resize(target_img, [1024, 1024], antialias=True)

        return input_img, target_img
