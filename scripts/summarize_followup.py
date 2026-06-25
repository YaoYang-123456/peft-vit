#!/usr/bin/env python3
"""汇总追加实验(results/followup/):
  表 A1  DTD 上五种方法对比(3 种子 mean±std)
  表 A2  DTD 上 LoRA/SSF 层位置消融 + 晚段 vs 其他的配对 t 检验
  表 B   Fisher 自动选层:每个数据集选中了哪些 Block,
         以及 lora-auto 与 all/early/even/late 各放法的对比
用法: python scripts/summarize_followup.py
(可在实验中途运行,缺失的配置会标记为 n=0 并跳过检验。)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "results", "summary.csv")
FUP = os.path.join(ROOT, "results", "followup")
FSUM = os.path.join(FUP, "summary.csv")


def load(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["method", "placement", "dataset", "seed", "test_acc"])
    return pd.read_csv(path).drop_duplicates(
        ["method", "placement", "dataset", "seed"], keep="last")


def acc(df, m, p, ds):
    g = df[(df.method == m) & (df.placement == p) & (df.dataset == ds)].sort_values("seed")
    return g["test_acc"].values


def fmt(v):
    if len(v) == 0:
        return "   --  (n=0)"
    if len(v) == 1:
        return f"{v.mean():6.2f}        (n=1)"
    return f"{v.mean():6.2f} ± {v.std(ddof=1):4.2f} (n={len(v)})"


def paired_p(a, b):
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    return stats.ttest_rel(a, b).pvalue


main = load(MAIN)
fup = load(FSUM)

print("=" * 68)
print("表 A1  DTD(47 类纹理)五种方法对比")
print("=" * 68)
for m in ["linear", "bitfit", "ssf", "lora", "full"]:
    print(f"  {m:<8} {fmt(acc(fup, m, 'all', 'dtd'))}")

print("\n" + "=" * 68)
print("表 A2  DTD 层位置消融(对照 §3.3/§3.4 的机制预测)")
print("=" * 68)
for m in ["lora", "ssf"]:
    base = acc(fup, m, "all", "dtd")
    print(f"  [{m}]  all-12: {fmt(base)}")
    late = acc(fup, m, "late", "dtd")
    for p in ["early", "mid", "even", "late"]:
        v = acc(fup, m, p, "dtd")
        pv = paired_p(v, late) if p != "late" else float("nan")
        extra = "" if p == "late" else f"   vs late: p={pv:.3f}" if pv == pv else ""
        print(f"    {p:<6} {fmt(v)}{extra}")

print("\n" + "=" * 68)
print("表 B  Fisher 自动选层(lora, r=8, k=4)")
print("=" * 68)
for ds in ["cifar100", "flowers", "dtd"]:
    sel = {}
    for f in sorted(glob.glob(os.path.join(FUP, f"fisher_{ds}_seed*.json"))):
        j = json.load(open(f))
        sel[j["seed"]] = j["selected"]
    if sel:
        print(f"\n  [{ds}] 各种子选中的 Block: " +
              "; ".join(f"seed{s}->{b}" for s, b in sorted(sel.items())))
    src = fup if ds == "dtd" else main          # 对照放法: dtd 在 followup, 其余在主结果
    auto = acc(fup, "lora", "auto", ds)
    print(f"    auto   {fmt(auto)}")
    for p in ["all", "early", "even", "late"]:
        v = acc(src, "lora", p, ds)
        pv = paired_p(auto, v)
        tail = f"   auto vs {p}: p={pv:.3f}" if pv == pv else ""
        print(f"    {p:<6} {fmt(v)}{tail}")

n_done = len(fup)
print(f"\n(进度: {n_done}/48 个 run 已完成; 缺失项会随实验推进自动补全)")
