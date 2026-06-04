"""Datasets, transforms and loaders.

Unified protocol (kept identical across all methods for fair comparison):
  - train: RandomResizedCrop(224, scale=(0.5,1.0)) + horizontal flip
  - eval : Resize(256) + CenterCrop(224)
  - normalization comes from the model's data config (here (0.5,0.5,0.5)).

Validation/test discipline:
  - Flowers-102 and DTD ship official train/val/test splits -> used directly.
  - CIFAR-100 and Pets have no official val split, so we carve a fixed,
    seeded validation set out of the training pool. The test set is only
    touched once, at the very end of training.
"""
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def to_rgb(img):
    """Some images (a few in Pets/DTD) are grayscale or RGBA; force 3-channel RGB."""
    return img.convert("RGB")


def build_transforms(mean, std, img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Lambda(to_rgb),
            transforms.RandomResizedCrop(
                img_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    resize = int(round(img_size / 0.875))  # 224 -> 256
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _split_train_val(n, val_ratio=0.1, seed=42):
    n_val = int(round(n * val_ratio))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(n), [n_train, n_val], generator=g)
    return list(tr), list(va)


def build_dataset(name, data_root, mean, std, img_size=224, val_ratio=0.1, seed=42):
    train_tf = build_transforms(mean, std, img_size, train=True)
    eval_tf = build_transforms(mean, std, img_size, train=False)
    name = name.lower()

    if name == "cifar100":
        full_tr = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
        full_ev = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=eval_tf)
        test = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=eval_tf)
        tr_idx, va_idx = _split_train_val(len(full_tr), val_ratio, seed)
        train, val = Subset(full_tr, tr_idx), Subset(full_ev, va_idx)
        num_classes = 100

    elif name == "flowers":
        train = torchvision.datasets.Flowers102(data_root, split="train", download=True, transform=train_tf)
        val = torchvision.datasets.Flowers102(data_root, split="val", download=True, transform=eval_tf)
        test = torchvision.datasets.Flowers102(data_root, split="test", download=True, transform=eval_tf)
        num_classes = 102

    elif name == "pets":
        full_tr = torchvision.datasets.OxfordIIITPet(
            data_root, split="trainval", target_types="category", download=True, transform=train_tf)
        full_ev = torchvision.datasets.OxfordIIITPet(
            data_root, split="trainval", target_types="category", download=True, transform=eval_tf)
        test = torchvision.datasets.OxfordIIITPet(
            data_root, split="test", target_types="category", download=True, transform=eval_tf)
        tr_idx, va_idx = _split_train_val(len(full_tr), val_ratio, seed)
        train, val = Subset(full_tr, tr_idx), Subset(full_ev, va_idx)
        num_classes = 37

    elif name == "dtd":
        train = torchvision.datasets.DTD(data_root, split="train", partition=1, download=True, transform=train_tf)
        val = torchvision.datasets.DTD(data_root, split="val", partition=1, download=True, transform=eval_tf)
        test = torchvision.datasets.DTD(data_root, split="test", partition=1, download=True, transform=eval_tf)
        num_classes = 47

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train, val, test, num_classes


def build_loaders(train, val, test, batch_size=128, num_workers=2):
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=2 * batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=2 * batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
