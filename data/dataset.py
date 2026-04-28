import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from .transforms import PairedTransform


# Split ranges per paper
SPLITS = {
    'train': (0, 55999),
    'val':   (56000, 62999),
    'test':  (63000, 69999),
}


def dilate(mask, radius=5):
    """Morphological dilation using max-pool."""
    pad = radius
    mask = F.pad(mask, [pad, pad, pad, pad], mode='constant', value=0)
    mask = F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=0)
    return mask


class FFHQRDataset(Dataset):
    """
    FFHQR dataset loader.

    Directory layout:
        original_dir/  — flat folder with {id:05d}.png (70k files)
        retouched_dir/ — subdirs {id//1000*1000:05d}/{id:05d}.png (1000 per subdir)

    Args:
        original_dir:  Path to originals (images1024x1024/)
        retouched_dir: Path to retouched (ffhqr/)
        split:         'train' | 'val' | 'test'
        patch_size:    Crop size during training (default 512)
        augment:       Enable augmentation (only for train)
    """

    def __init__(self, original_dir, retouched_dir, split='train',
                 patch_size=512, augment=None):
        assert split in SPLITS, f"split must be one of {list(SPLITS.keys())}"
        self.original_dir = original_dir
        self.retouched_dir = retouched_dir
        self.split = split

        if augment is None:
            augment = (split == 'train')

        self.transform = PairedTransform(patch_size=patch_size, augment=augment)

        start, end = SPLITS[split]
        # Only include IDs where both original and retouched exist
        self.ids = []
        for i in range(start, end + 1):
            orig_path = os.path.join(original_dir, f'{i:05d}.png')
            subdir = f'{(i // 1000) * 1000:05d}'
            ret_path = os.path.join(retouched_dir, subdir, f'{i:05d}.png')
            if os.path.exists(orig_path) and os.path.exists(ret_path):
                self.ids.append(i)

        print(f"[FFHQRDataset] split={split}, found {len(self.ids)} pairs "
              f"(range {start}-{end})")

    def __len__(self):
        return len(self.ids)

    def _get_paths(self, idx):
        i = self.ids[idx]
        orig_path = os.path.join(self.original_dir, f'{i:05d}.png')
        subdir = f'{(i // 1000) * 1000:05d}'
        ret_path = os.path.join(self.retouched_dir, subdir, f'{i:05d}.png')
        return orig_path, ret_path, f'{i:05d}'

    def __getitem__(self, idx):
        orig_path, ret_path, name = self._get_paths(idx)

        # Read as float [0, 1]
        input_img = read_image(orig_path, mode=ImageReadMode.RGB).float() / 255.0
        target_img = read_image(ret_path, mode=ImageReadMode.RGB).float() / 255.0

        # Apply paired transforms
        input_img, target_img = self.transform(input_img, target_img)

        # Derive blemish mask from diff
        diff = torch.abs(target_img - input_img)
        gray = diff.mean(dim=0, keepdim=True)             # [1, H, W]
        mask = (gray > 0.02).float()
        mask = dilate(mask.unsqueeze(0), radius=5).squeeze(0)  # [1, H, W]
        mask = mask.expand(3, -1, -1)                      # [3, H, W]

        return {
            'input': input_img,
            'target': target_img,
            'mask': mask,
            'name': name,
        }
