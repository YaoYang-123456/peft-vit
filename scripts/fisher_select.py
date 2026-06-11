#!/usr/bin/env python3
"""Fisher-guided automatic layer selection for PEFT (§"深度感知 PEFT"的自动化版本).

动机:§3.3 发现"固定预算放在哪些 Block"在分布偏移数据(CIFAR-100)上影响显著,
但 early/mid/even 这类放法仍是人工指定的。本脚本给出一个数据驱动的一次性选层准则:

  1. 冻结整个主干,只快速训练分类头若干步(线性探测式 warmup),
     使梯度信号与任务对齐(随机头的梯度近似噪声);
  2. 打开每个 Block 的 attn.qkv.weight 的梯度(正是 LoRA 的注入位置),
     在若干 mini-batch 上累积经验 Fisher 信息(梯度平方和),但不更新任何参数;
  3. 每个 Block 的得分 = 其 qkv 权重的 Fisher 总量(各 Block 形状相同,可直接比较),
     取得分最高的 k 个 Block 作为 LoRA 的放置位置。

直觉:Fisher 大 = 该层权重对目标任务损失最"敏感"、最需要被适配。
预测:在 CIFAR-100(低分辨率偏移)上应选中早/中层并匹敌 early/even 放法;
在 Flowers 等与预训练分布接近的数据上,选哪里都差不多(与 §3.4 机制一致)。

用法:
  python scripts/fisher_select.py --dataset cifar100 --seed 42 --k 4 \
      --data-root ./data --out-dir ./results/followup
输出:
  <out-dir>/fisher_<dataset>_seed<seed>.json   (含 12 个 Block 的得分与选中序号)
  并在 stdout 打印  SELECTED_BLOCKS=0,2,3,5  供 shell 脚本读取。

整个过程只前向/反向、不训练主干,在单卡上约 2–5 分钟。
"""
import argparse
import json
import os
import sys
import time

# 该脚本位于 scripts/ 下,需把项目根目录加入 sys.path 才能 import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.utils import set_seed, get_device
from src.backbone import create_model, get_default_data_config, DEFAULT_MODEL
from src.data import build_dataset, build_loaders


def cycle(loader):
    """无限循环一个 DataLoader(小数据集一个 epoch 不足以凑够步数时使用)。"""
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-dir", default="./results/followup")
    ap.add_argument("--k", type=int, default=4, help="选出的 Block 数(与层位置消融一致取 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=300,
                    help="分类头 warmup 步数(线性探测式,仅头部可训练)")
    ap.add_argument("--head-lr", type=float, default=1e-2)
    ap.add_argument("--fisher-batches", type=int, default=200,
                    help="累积经验 Fisher 的 mini-batch 数")
    ap.add_argument("--backbone", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    t0 = time.time()

    model_name = args.backbone or DEFAULT_MODEL
    dcfg = get_default_data_config(model_name)
    train_set, val_set, test_set, num_classes = build_dataset(
        args.dataset, args.data_root, dcfg["mean"], dcfg["std"],
        img_size=224, val_ratio=0.1, seed=args.seed)
    train_loader, _, _ = build_loaders(train_set, val_set, test_set,
                                       args.batch_size, args.num_workers)

    model = create_model(num_classes=num_classes, drop_path_rate=0.0,
                         model_name=model_name).to(device)
    model.eval()  # 无 dropout/drop_path,eval() 仅为确定性;梯度仍正常计算

    # ---- 阶段 1:冻结主干,只快速训练分类头(使梯度与任务对齐)----
    for p in model.parameters():
        p.requires_grad_(False)
    head_params = list(model.get_classifier().parameters())
    for p in head_params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    it = cycle(train_loader)
    for step in range(args.warmup_steps):
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            print(f"[head warmup] step {step + 1}/{args.warmup_steps}  loss={loss.item():.3f}")

    # ---- 阶段 2:在每个 Block 的 qkv.weight 上累积经验 Fisher(不更新参数)----
    for p in head_params:
        p.requires_grad_(False)
    n_blocks = len(model.blocks)
    for blk in model.blocks:
        blk.attn.qkv.weight.requires_grad_(True)

    scores = [0.0] * n_blocks
    it = cycle(train_loader)
    for b in range(args.fisher_batches):
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = criterion(model(x), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for i, blk in enumerate(model.blocks):
            g = blk.attn.qkv.weight.grad
            if g is not None:
                scores[i] += g.detach().float().pow(2).sum().item()
        if (b + 1) % 50 == 0:
            print(f"[fisher] batch {b + 1}/{args.fisher_batches}")
    model.zero_grad(set_to_none=True)

    total = sum(scores) or 1.0
    norm = [s / total for s in scores]
    selected = sorted(sorted(range(n_blocks), key=lambda i: scores[i], reverse=True)[:args.k])

    print("\nPer-block Fisher share (qkv.weight):")
    for i in range(n_blocks):
        bar = "#" * int(round(norm[i] * 60))
        mark = "  <== selected" if i in selected else ""
        print(f"  block {i:2d}  {norm[i] * 100:5.1f}%  {bar}{mark}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"fisher_{args.dataset}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(dict(dataset=args.dataset, seed=args.seed, k=args.k,
                       backbone=model_name,
                       warmup_steps=args.warmup_steps,
                       fisher_batches=args.fisher_batches,
                       scores=scores, score_share=norm, selected=selected,
                       minutes=round((time.time() - t0) / 60, 1)), f, indent=2)
    print(f"\nwrote {out_path}  ({(time.time() - t0) / 60:.1f} min)")
    print("SELECTED_BLOCKS=" + ",".join(map(str, selected)))


if __name__ == "__main__":
    main()
