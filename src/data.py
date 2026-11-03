"""Dataset loading for the apple-scab binary classifier."""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from src import CLASSES

IMG_SIZE = 224


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class LeafDataset(Dataset):
    """ImageFolder-style dataset restricted to the two classes in src.CLASSES.

    Pass resplit=False when root is already a single split folder (e.g. the
    Kaggle dataset's own train/ or valid/ folder) — then every image in it is
    used as-is. With resplit=True (default), root is one folder holding both
    classes and it is randomly split split_ratio / (1 - split_ratio) into
    the train or val side.
    """

    def __init__(self, root, split="train", split_ratio=0.85, seed=42,
                 transform=None, resplit=True):
        self.root = Path(root)
        self.transform = transform
        items = []
        for label, cls in enumerate(CLASSES):
            for p in sorted((self.root / cls).rglob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    items.append((str(p), label))
        if resplit:
            rng = random.Random(seed)
            rng.shuffle(items)
            cut = int(len(items) * split_ratio)
            self.items = items[:cut] if split == "train" else items[cut:]
        else:
            self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(data_root, val_root=None, batch=32, workers=4, seed=42):
    train_tf, eval_tf = build_transforms()
    if val_root is not None:
        # The Kaggle kernel trains on the dataset's own train/valid folders;
        # pass them here to reproduce the trained model exactly.
        train_ds = LeafDataset(data_root, resplit=False, transform=train_tf)
        val_ds = LeafDataset(val_root, resplit=False, transform=eval_tf)
    else:
        train_ds = LeafDataset(data_root, split="train", seed=seed, transform=train_tf)
        val_ds = LeafDataset(data_root, split="val", seed=seed, transform=eval_tf)

    # Balance scab vs healthy with weighted sampling. The Kaggle "Augmented"
    # split used for training is already near-balanced (2,016 scab / 2,008
    # healthy), so the sampler is close to uniform there; it matters more for
    # the raw PlantVillage counts (~630 / ~1,645).
    labels = [lbl for _, lbl in train_ds.items]
    counts = torch.bincount(torch.tensor(labels)).float()
    weights = 1.0 / counts
    sampler = WeightedRandomSampler([float(weights[l]) for l in labels],
                                    num_samples=len(labels), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, sampler=sampler,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch * 2, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader, val_ds
