#!/usr/bin/env python3
"""从 results/summary.csv 自动再生成论文里所有派生数据,确保完全可复现:

  - main_comparison.csv     表 2 (五方法 × 三数据集, 3 种子 mean/std/n)
  - placement_ablation.csv  表 3/4 (LoRA/SSF 层位置, 各数据集 mean±std)
  - significance.csv         配对 t 检验 p 值 (报告 §3.3/§3.4 引用)
  - efficiency.csv          表 6 (峰值显存=各种子一致值; 每轮时间=seed42 代表性运行)

用法: python scripts/summarize_results.py
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RUNS = os.path.join(RES, "runs")

# 若 runs/ 不存在但有 runs.zip(为方便上传 GitHub,逐轮日志打包存放),自动解压。
# efficiency.csv 的每轮时间需要逐轮日志,没有这步会被写成 NaN。
import zipfile
if not os.path.isdir(RUNS) and os.path.exists(os.path.join(RES, "runs.zip")):
    with zipfile.ZipFile(os.path.join(RES, "runs.zip")) as _z:
        _z.extractall(RES)

raw = pd.read_csv(os.path.join(RES, "summary.csv"))
raw = raw.drop_duplicates(["method", "placement", "dataset", "seed"], keep="last")

DSS = ["cifar100", "flowers", "pets"]
METHODS = ["linear", "bitfit", "ssf", "lora", "full"]
PLACES = ["all", "early", "mid", "even", "late"]


def vec(method, placement, ds):
    g = raw[(raw.method == method) & (raw.placement == placement) & (raw.dataset == ds)].sort_values("seed")
    return g["test_acc"].values


# --------------------------------------------------------------------------- #
#  1) main_comparison.csv  (表 2)
# --------------------------------------------------------------------------- #
rows = []
for m in METHODS:
    row = {"method": m, "trainable": int(vec_tr := raw[(raw.method == m) & (raw.placement == "all") &
                                                       (raw.dataset == "cifar100")]["trainable"].iloc[0])}
    for ds in DSS:
        v = vec(m, "all", ds)
        row[f"{ds}_mean"] = round(float(np.mean(v)), 2)
        row[f"{ds}_std"] = round(float(np.std(v, ddof=1)), 2)
        row[f"{ds}_n"] = int(len(v))
    rows.append(row)
main = pd.DataFrame(rows)
main.to_csv(os.path.join(RES, "main_comparison.csv"), index=False)
print("== main_comparison.csv (表 2) ==")
print(main.to_string(index=False), "\n")

# --------------------------------------------------------------------------- #
#  2) placement_ablation.csv  (表 3 = cifar100; 表 4 = flowers/pets)
# --------------------------------------------------------------------------- #
ab = []
for ds in DSS:
    for method in ["lora", "ssf"]:
        for p in PLACES:
            v = vec(method, p, ds)
            if len(v) == 0:
                continue
            ab.append(dict(dataset=ds, method=method, placement=p,
                           trainable=int(raw[(raw.method == method) & (raw.placement == p) &
                                             (raw.dataset == ds)]["trainable"].iloc[0]),
                           test_acc=round(float(np.mean(v)), 2),
                           test_std=round(float(np.std(v, ddof=1)), 2), n=len(v)))
abl = pd.DataFrame(ab)
abl.to_csv(os.path.join(RES, "placement_ablation.csv"), index=False)

# --------------------------------------------------------------------------- #
#  3) significance.csv  (paired t-tests over the 3 seeds)
# --------------------------------------------------------------------------- #
sig = []
for ds in DSS:
    for method in ["lora", "ssf"]:
        for a, b in combinations(PLACES, 2):
            va, vb = vec(method, a, ds), vec(method, b, ds)
            if len(va) < 2 or len(vb) < 2 or len(va) != len(vb):
                continue
            t, p = stats.ttest_rel(va, vb)
            sig.append(dict(dataset=ds, method=method, comparison=f"{a}_vs_{b}",
                            mean_diff=round(float(np.mean(va) - np.mean(vb)), 3),
                            p_value=round(float(p), 4), significant=bool(p < 0.05)))
sigdf = pd.DataFrame(sig)
sigdf.to_csv(os.path.join(RES, "significance.csv"), index=False)

print("== 配对 t 检验: CIFAR-100 '晚段 vs 其它' (报告 §3.3 主结论) ==")
sub = sigdf[(sigdf.dataset == "cifar100") & (sigdf.comparison.str.contains("late"))]
print(sub.to_string(index=False), "\n")

print("== 配对 t 检验: Flowers/Pets 上达到未校正 p<0.05 的对 (报告 §3.4) ==")
hi = sigdf[(sigdf.dataset.isin(["flowers", "pets"])) & (sigdf.significant)]
print((hi.to_string(index=False) if len(hi) else "  (无)"), "\n")

# --------------------------------------------------------------------------- #
#  4) efficiency.csv  (表 6)
#     peak memory is deterministic (identical across seeds); per-epoch wall time
#     is taken from the seed-42 run as a representative figure (cross-seed wall
#     time varies up to ~2x with GPU load, so a mean would be misleading).
# --------------------------------------------------------------------------- #
eff = []
for m in METHODS:
    g = raw[(raw.method == m) & (raw.placement == "all") & (raw.dataset == "cifar100")]
    mem = int(round(g["peak_mem_mb"].iloc[0]))           # consistent across seeds
    f = os.path.join(RUNS, f"{m}_cifar100_seed42.csv")
    spe = round(float(pd.read_csv(f)["time_s"].mean()), 1) if os.path.exists(f) else np.nan
    eff.append(dict(method=m, trainable=int(g["trainable"].iloc[0]),
                    pct=round(float(g["pct"].iloc[0]), 3), peak_mem_MB=mem, sec_per_epoch=spe))
effdf = pd.DataFrame(eff)
effdf.to_csv(os.path.join(RES, "efficiency.csv"), index=False)
print("== efficiency.csv (表 6; 显存=各种子一致值, 时间=seed42 代表性运行) ==")
print(effdf.to_string(index=False))
