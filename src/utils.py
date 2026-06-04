"""Small reusable utilities: reproducibility, metrics, logging."""
import os
import csv
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Make a run reproducible (as far as cuDNN allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def correct_count(logits, target, topk=(1,)):
    """Number of correct predictions in the batch for each k in topk."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)        # (B, maxk)
    pred = pred.t()                            # (maxk, B)
    hit = pred.eq(target.view(1, -1).expand_as(pred))
    return [hit[:k].reshape(-1).float().sum().item() for k in topk]


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)


class CSVLogger:
    """Writes one row per epoch to a CSV file."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def append_summary(path, row):
    """Append a one-line run summary to a shared results table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def cpu_state_dict(model):
    """A detached CPU copy of the model state (for keeping the best-val checkpoint)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
