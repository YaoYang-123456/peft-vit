"""Train one (method, dataset) configuration.

Examples
--------
python train.py --method linear --dataset flowers --epochs 100
python train.py --method full   --dataset cifar100 --epochs 50
"""
import argparse
import os
import time

import torch

from src.utils import (set_seed, get_device, count_parameters,
                       CSVLogger, append_summary, cpu_state_dict)
from src.backbone import create_model, get_default_data_config, DEFAULT_MODEL
from src.data import build_dataset, build_loaders
from src.methods import configure_method
from src.engine import train_one_epoch, evaluate


# Per-method training defaults. PEFT methods use a much higher LR than full FT,
# and only full FT benefits from stochastic depth (drop_path).
METHOD_DEFAULTS = {
    "linear":      dict(lr=1e-2, weight_decay=1e-4, drop_path=0.0, optimizer="adamw"),
    "full":        dict(lr=1e-5, weight_decay=5e-2, drop_path=0.1, optimizer="adamw"),
    # ---- PEFT methods (implemented in methods.py) ----
    "bitfit":      dict(lr=1e-3, weight_decay=1e-4, drop_path=0.0, optimizer="adamw"),
    "lora":        dict(lr=1e-3, weight_decay=1e-4, drop_path=0.0, optimizer="adamw"),
    "ssf":         dict(lr=1e-3, weight_decay=5e-4, drop_path=0.0, optimizer="adamw"),
}


def parse_method(method):
    """Split a method string into (base, placement). Examples:
    "lora" -> ("lora","all");  "lora-early" -> ("lora","early");
    "lora-r12-early" -> ("lora","early")  (rank handled inside methods.py)."""
    toks = method.lower().split("-")
    base, placement = toks[0], "all"
    for t in toks[1:]:
        if not (len(t) > 1 and t[0] == "r" and t[1:].isdigit()):
            placement = t
    return base, placement


def build_optimizer(model, name, lr, weight_decay):
    """AdamW/SGD over trainable params; no weight decay on biases/1-D (norm) params."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None,
                    help="override the method-default learning rate")
    ap.add_argument("--wd", "--weight-decay", dest="wd", type=float, default=None,
                    help="override the method-default weight decay")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--subset-size", type=int, default=None,
                    help="use only the first N training samples (quick code check)")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save best-on-val trainable slice (adapters+head) to "
                         "<out-dir>/runs/<run>_adapter.pt")
    ap.add_argument("--backbone", type=str, default=None,
                    help="timm 主干名;默认 ViT-B/16 IN-21k (vit_base_patch16_224.augreg_in21k)")
    ap.add_argument("--blocks", type=str, default=None,
                    help="逗号分隔的 Block 序号(如 '0,3,5,7'),仅对 lora/ssf 生效;"
                         "给出时覆盖 method 中的位置后缀(用于自动选层实验)")
    ap.add_argument("--placement-label", type=str, default=None,
                    help="与 --blocks 搭配使用,写入 summary.csv 的 placement 标签(如 'auto')")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = (not args.no_amp) and (device.type == "cuda")

    base_method, placement = parse_method(args.method)
    explicit_blocks = None
    if args.blocks is not None:
        explicit_blocks = sorted({int(b) for b in args.blocks.split(",") if b.strip() != ""})
        placement = args.placement_label or "custom"
    d = dict(METHOD_DEFAULTS[base_method])
    if args.lr is not None: d["lr"] = args.lr
    if args.wd is not None: d["weight_decay"] = args.wd
    lr, wd = d["lr"], d["weight_decay"]
    drop_path, opt_name = d["drop_path"], d["optimizer"]

    # 主干:默认 ViT-B/16 IN-21k;--backbone 可换(如 vit_small_patch16_224.augreg_in21k)
    model_name = args.backbone or DEFAULT_MODEL
    # data preprocessing config (no weight download) —— 归一化/尺寸随主干自动解析
    dcfg = get_default_data_config(model_name)
    mean, std = dcfg["mean"], dcfg["std"]

    train_set, val_set, test_set, num_classes = build_dataset(
        args.dataset, args.data_root, mean, std,
        img_size=224, val_ratio=args.val_ratio, seed=args.seed)
    if args.subset_size is not None:
        from torch.utils.data import Subset
        n = min(args.subset_size, len(train_set))
        train_set = Subset(train_set, list(range(n)))
        print(f"[quick mode] using only {n} training samples")
    train_loader, val_loader, test_loader = build_loaders(
        train_set, val_set, test_set, args.batch_size, args.num_workers)

    model = create_model(num_classes=num_classes, drop_path_rate=drop_path, model_name=model_name).to(device)
    if explicit_blocks is not None:
        n_blk = len(model.blocks)
        bad = [b for b in explicit_blocks if not (0 <= b < n_blk)]
        if bad:
            raise ValueError(f"--blocks 中的序号 {bad} 超出范围 [0, {n_blk - 1}]")
        print(f"[explicit blocks] adapting blocks {explicit_blocks} (label='{placement}')")
    configure_method(model, args.method, block_ids=explicit_blocks)
    trainable, total = count_parameters(model)

    print(f"[{args.method} | {args.dataset}] classes={num_classes} | "
          f"trainable={trainable:,} ({100.0 * trainable / total:.3f}% of {total:,})")
    print(f"lr={lr}  wd={wd}  drop_path={drop_path}  opt={opt_name}  "
          f"epochs={args.epochs}  bs={args.batch_size}  amp={use_amp}  device={device}")

    optimizer = build_optimizer(model, opt_name, lr, wd)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)
    run_name = f"{args.method}_{args.dataset}_seed{args.seed}"
    if explicit_blocks is not None:
        run_name = f"{base_method}-{placement}_{args.dataset}_seed{args.seed}"
    logger = CSVLogger(
        os.path.join(run_dir, run_name + ".csv"),
        ["epoch", "train_loss", "train_acc", "val_acc", "lr", "time_s", "peak_mem_mb"])

    best_val, best_state = -1.0, None
    peak_run_mb, t_start = 0.0, time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, cur_lr = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            total_steps, warmup_steps, lr,
            step_offset=epoch * steps_per_epoch, use_amp=use_amp)
        val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        dt = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        peak_run_mb = max(peak_run_mb, peak_mb)

        logger.log(dict(epoch=epoch + 1, train_loss=round(tr_loss, 4),
                        train_acc=round(tr_acc, 3), val_acc=round(val_acc, 3),
                        lr=round(cur_lr, 6), time_s=round(dt, 1), peak_mem_mb=round(peak_mb, 1)))
        print(f"epoch {epoch + 1:3d}/{args.epochs} | loss {tr_loss:.3f} | "
              f"train {tr_acc:5.2f}% | val {val_acc:5.2f}% | {dt:5.1f}s | {peak_mb:6.0f}MB")

        if val_acc > best_val:
            best_val = val_acc
            best_state = cpu_state_dict(model)

    train_time_s = round(time.time() - t_start, 1)
    # final TEST with the best-on-validation weights (test touched exactly once)
    if best_state is not None:
        model.load_state_dict(best_state)
        if args.save_ckpt:
            # only the trainable slice (adapters + head) is needed to reproduce a
            # PEFT run -- this is the whole point of PEFT (store a few % per task).
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            slim = {k: v for k, v in best_state.items() if k in trainable_names}
            torch.save(slim, os.path.join(run_dir, run_name + "_adapter.pt"))
    test_acc = evaluate(model, test_loader, device, use_amp=use_amp)

    print(f"\n== {run_name}: best_val={best_val:.2f}%  test={test_acc:.2f}%  "
          f"trainable={trainable:,} ({100.0 * trainable / total:.3f}%)")

    append_summary(os.path.join(args.out_dir, "summary.csv"), dict(
        method=base_method, placement=placement, dataset=args.dataset, seed=args.seed,
        trainable=trainable, total=total, pct=round(100.0 * trainable / total, 4),
        best_val=round(best_val, 3), test_acc=round(test_acc, 3),
        epochs=args.epochs, lr=lr, wd=wd, drop_path=drop_path,
        peak_mem_mb=round(peak_run_mb, 1), train_time_s=train_time_s))


if __name__ == "__main__":
    main()
