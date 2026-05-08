import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

class SODDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        image_size: int = 128,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.root_dir   = Path(root_dir)
        self.image_size = image_size
        self.augment    = augment
        self.normalize  = normalize

        # Collect paired paths
        img_dir  = self.root_dir / "images"
        mask_dir = self.root_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Expected 'images/' and 'masks/' subdirectories inside '{root_dir}'. "
                "Please organise your dataset accordingly."
            )

        img_paths = {p.stem: p for p in img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}

        self.pairs = []
        for mask_path in sorted(mask_dir.iterdir()):
            stem = mask_path.stem
            if stem in img_paths:
                self.pairs.append((img_paths[stem], mask_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No matched image-mask pairs found. "
                "Ensure filenames (without extension) match between images/ and masks/."
            )

        print(f"[SODDataset] Loaded {len(self.pairs)} pairs from '{root_dir}'  "
              f"| size={image_size}  augment={augment}")

    def _resize(self, img: Image.Image, mask: Image.Image):
        img  = TF.resize(img,  [self.image_size, self.image_size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.NEAREST)
        return img, mask

    def _augment(self, img: Image.Image, mask: Image.Image):

        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.7:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            i, j, h, w = T.RandomCrop.get_params(
                img, output_size=(int(self.image_size * 0.8), int(self.image_size * 0.8))
            )
            img  = TF.resized_crop(img,  i, j, h, w,
                                   [self.image_size, self.image_size],
                                   interpolation=Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w,
                                   [self.image_size, self.image_size],
                                   interpolation=Image.NEAREST)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img  = TF.rotate(img,  angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        if random.random() > 0.5:
            img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img,   contrast_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            img = TF.adjust_saturation(img, saturation_factor=random.uniform(0.7, 1.3))

        return img, mask

    def _to_tensor(self, img: Image.Image, mask: Image.Image):
        img_t  = TF.to_tensor(img) 
        mask_t = TF.to_tensor(mask.convert("L")) 
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t

    def _normalize(self, img_t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_t - mean) / std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img, mask = self._resize(img, mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t, mask_t = self._to_tensor(img, mask)

        if self.normalize:
            img_t = self._normalize(img_t)

        return img_t, mask_t, str(img_path.name)

def build_dataloaders(
    root_dir: str,
    image_size:    int   = 128,
    batch_size:    int   = 16,
    num_workers:   int   = 2,
    train_ratio:   float = 0.70,
    val_ratio:     float = 0.15,
    seed:          int   = 42,
):

    full_ds = SODDataset(root_dir, image_size=image_size, augment=False, normalize=True)

    total    = len(full_ds)
    n_train  = int(total * train_ratio)
    n_val    = int(total * val_ratio)
    n_test   = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test],
                                              generator=generator)

    train_aug = _AugmentedSubset(full_ds, train_ds.indices, image_size=image_size)

    print(f"\n[DataLoaders] train={len(train_aug)}  val={len(val_ds)}  test={len(test_ds)}")

    loaders = {
        "train": DataLoader(train_aug, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "full_dataset": full_ds,
    }
    return loaders


class _AugmentedSubset(Dataset):

    def __init__(self, base_dataset: SODDataset, indices, image_size: int):
        self.base     = base_dataset
        self.indices  = indices
        self.aug_ds   = SODDataset(
            str(base_dataset.root_dir),
            image_size=image_size,
            augment=True,
            normalize=True,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.aug_ds[self.indices[idx]]

def visualise_batch(loader, num_samples: int = 4, save_path: str = None):
    imgs, masks, names = next(iter(loader))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 3))
    if num_samples == 1:
        axes = [axes]

    for i in range(min(num_samples, len(imgs))):
        img_disp = (imgs[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        msk_disp = masks[i].squeeze().numpy()

        axes[i][0].imshow(img_disp)
        axes[i][0].set_title(f"Image: {names[i][:20]}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(msk_disp, cmap="gray")
        axes[i][1].set_title("GT Mask", fontsize=8)
        axes[i][1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualise_batch] Saved → {save_path}")
    else:
        plt.show()
    plt.close()

def generate_synthetic_dataset(root_dir: str, n_samples: int = 200, image_size: int = 128):

    root = Path(root_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n_samples):
        img_arr  = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)

        mask_arr = np.zeros((image_size, image_size), dtype=np.uint8)
        cx = rng.integers(image_size // 4, 3 * image_size // 4)
        cy = rng.integers(image_size // 4, 3 * image_size // 4)
        r  = rng.integers(image_size // 8, image_size // 3)
        ys, xs = np.ogrid[:image_size, :image_size]
        mask_arr[(xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2] = 255

        Image.fromarray(img_arr).save(root / "images" / f"sample_{i:04d}.jpg")
        Image.fromarray(mask_arr).save(root / "masks"  / f"sample_{i:04d}.png")

    print(f"[generate_synthetic_dataset] Created {n_samples} samples in '{root_dir}'")

if __name__ == "__main__":
    SYNTH_DIR = "data/synthetic"
    generate_synthetic_dataset(SYNTH_DIR, n_samples=100, image_size=128)

    loaders = build_dataloaders(SYNTH_DIR, image_size=128, batch_size=8)

    print("\nBatch shapes:")
    imgs, masks, names = next(iter(loaders["train"]))
    print(f"  images : {imgs.shape}   dtype={imgs.dtype}")
    print(f"  masks  : {masks.shape}  dtype={masks.dtype}")
    print(f"  range  : [{imgs.min():.2f}, {imgs.max():.2f}]")

    visualise_batch(loaders["train"], num_samples=4, save_path="sample_batch.png")
    print("\ndata_loader.py self-test passed ✓")