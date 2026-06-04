"""Training/evaluation loops with mixed precision and a per-step warmup+cosine LR."""
import math

import torch
import torch.nn as nn

from .utils import AverageMeter, correct_count


def adjust_learning_rate(optimizer, step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    """Linear warmup then cosine decay, applied every optimization step."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, scaler, device,
                    total_steps, warmup_steps, base_lr, step_offset, use_amp=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    correct, total, cur_lr = 0, 0, base_lr

    for i, (x, y) in enumerate(loader):
        cur_lr = adjust_learning_rate(optimizer, step_offset + i, total_steps, warmup_steps, base_lr)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        correct += correct_count(logits.detach(), y, (1,))[0]
        total += x.size(0)

    return loss_meter.avg, 100.0 * correct / max(1, total), cur_lr


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        correct += correct_count(logits, y, (1,))[0]
        total += x.size(0)
    return 100.0 * correct / max(1, total)
