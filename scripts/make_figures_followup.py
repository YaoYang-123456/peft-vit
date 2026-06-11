#!/usr/bin/env python3
"""生成追加实验的两张图与显著性表(从 results/followup/ 原始数据,可复现):
  fig6_dtd_placement.png   DTD 层位置消融(LoRA 晚层反而最优)
  fig9_fisher_auto.png     Fisher 自动选层:逐块得分分布 + auto 与各放法对比
  followup_significance.csv 追加实验的配对 t 检验汇总
用法: python scripts/make_figures_followup.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FUP = os.path.join(RES, "followup")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

main = pd.read_csv(os.path.join(RES, "summary.csv")).drop_duplicates(
    ["method", "placement", "dataset", "seed"], keep="last")
fup = pd.read_csv(os.path.join(FUP, "summary.csv")).drop_duplicates(
    ["method", "placement", "dataset", "seed"], keep="last")


def acc(df, m, p, ds):
    return df[(df.method == m) & (df.placement == p) & (df.dataset == ds)
              ].sort_values("seed")["test_acc"].values


# ---------------- fig6: DTD placement ----------------
places = ["early", "mid", "even", "late"]
xl = ["early\n(0–3)", "mid\n(4–7)", "even\n(0,3,6,9)", "late\n(8–11)"]
x = np.arange(4)
w = 0.38
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for off, m, color in [(-w / 2, "lora", "#3b6fb6"), (w / 2, "ssf", "#e8a33d")]:
    mus = [acc(fup, m, p, "dtd").mean() for p in places]
    sds = [acc(fup, m, p, "dtd").std(ddof=1) for p in places]
    ax.bar(x + off, mus, w, yerr=sds, capsize=4, color=color,
           edgecolor="black", lw=0.4,
           label=f"{'LoRA' if m == 'lora' else 'SSF'} (4 blocks)")
    for xi, mu in zip(x + off, mus):
        ax.text(xi, mu + 0.06, f"{mu:.1f}", ha="center", va="bottom", fontsize=8)
    allv = acc(fup, m, "all", "dtd").mean()
    ax.axhline(allv, color=color, ls="--", lw=1.2, alpha=0.9)
    ax.text(3.45, allv + 0.04, f"{'LoRA' if m == 'lora' else 'SSF'} all-12 ({allv:.2f})",
            color=color, fontsize=8, ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(xl)
ax.set_ylim(75.8, 80.4)
ax.set_xlabel("Which 4 transformer blocks are adapted")
ax.set_ylabel("DTD test accuracy (%)  (mean±std, 3 seeds)")
ax.set_title("DTD (texture: semantic shift, natural low-level statistics):\n"
             "for LoRA the pattern REVERSES — late placement matches all-12")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_dtd_placement.png"), dpi=150)
plt.close(fig)

# ---------------- fig9: Fisher auto-selection ----------------
DS = ["cifar100", "flowers", "dtd"]
DSN = {"cifar100": "CIFAR-100", "flowers": "Flowers-102", "dtd": "DTD"}
COL = {"cifar100": "#3b6fb6", "flowers": "#4d9e4d", "dtd": "#c0392b"}
shares, sels = {}, {}
for ds in DS:
    rows, ss = [], []
    for f in sorted(glob.glob(os.path.join(FUP, f"fisher_{ds}_seed*.json"))):
        j = json.load(open(f))
        rows.append(np.array(j["score_share"]) * 100)
        ss.append(tuple(j["selected"]))
    shares[ds] = np.vstack(rows)
    sels[ds] = ss

fig = plt.figure(figsize=(12.6, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

axa = fig.add_subplot(gs[0, :])
blocks = np.arange(12)
for ds in DS:
    mu, sd = shares[ds].mean(0), shares[ds].std(0, ddof=1)
    axa.plot(blocks, mu, "-o", ms=4.5, lw=1.8, color=COL[ds], label=DSN[ds])
    axa.fill_between(blocks, mu - sd, mu + sd, color=COL[ds], alpha=0.15)
# mark the (modal) selected blocks for the two stable cases
from collections import Counter
for ds, dy in [("cifar100", 0.55), ("dtd", -0.75)]:
    modal = Counter(sels[ds]).most_common(1)[0][0]
    mu = shares[ds].mean(0)
    axa.scatter(list(modal), mu[list(modal)] + 0, s=150, facecolors="none",
                edgecolors=COL[ds], lw=1.8, zorder=5)
axa.set_xticks(blocks)
axa.set_xlabel("Transformer block index")
axa.set_ylabel("Empirical Fisher share of qkv.weight (%)")
axa.set_title("(a) Where the gradient signal lives: per-block Fisher share "
              "(mean±std over 3 seeds; circles = blocks picked by the criterion)\n"
              "CIFAR-100: early-heavy & stable    |    Flowers: flat/noisy (selection unstable)    |    "
              "DTD: mid-peaked — yet the BEST placement is late",
              fontsize=10)
axa.legend(loc="upper right", fontsize=9)
axa.grid(alpha=0.3)

order = ["early", "even", "late", "all", "auto"]
for k, ds in enumerate(DS):
    ax = fig.add_subplot(gs[1, k])
    src_pl = fup if ds == "dtd" else main
    mus, sds, cols = [], [], []
    for p in order:
        v = acc(fup if p == "auto" or ds == "dtd" else main, "lora", p, ds)
        mus.append(v.mean())
        sds.append(v.std(ddof=1))
        cols.append("#c0392b" if p == "auto" else "#9ecae1")
    bars = ax.bar(np.arange(len(order)), mus, 0.62, yerr=sds, capsize=3,
                  color=cols, edgecolor="black", lw=0.4)
    for xi, mu in enumerate(mus):
        ax.text(xi, mu + sds[xi] + 0.03, f"{mu:.1f}", ha="center",
                va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)
    pad = {"cifar100": (90.3, 93.3), "flowers": (98.6, 99.75), "dtd": (76.3, 80.2)}[ds]
    ax.set_ylim(*pad)
    ax.set_title(f"(b{k + 1}) LoRA on {DSN[ds]}", fontsize=10)
    if k == 0:
        ax.set_ylabel("Test accuracy (%)")
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("Fisher-guided automatic layer selection: succeeds under low-level shift "
             "(CIFAR-100), harmless without shift (Flowers), fails under semantic shift (DTD)",
             fontsize=11.5, y=0.99)
fig.savefig(os.path.join(FIG, "fig9_fisher_auto.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- significance CSV ----------------
rows = []
def add(tag, a, b):
    if len(a) == len(b) and len(a) >= 2:
        rows.append(dict(comparison=tag, mean_a=round(a.mean(), 2),
                         mean_b=round(b.mean(), 2),
                         p_paired_t=round(stats.ttest_rel(a, b).pvalue, 4)))

for m in ["lora", "ssf"]:
    late = acc(fup, m, "late", "dtd")
    for p in ["early", "mid", "even", "all"]:
        add(f"dtd:{m}:late_vs_{p}", late, acc(fup, m, p, "dtd"))
add("dtd:lora_all_vs_ssf_all", acc(fup, "lora", "all", "dtd"), acc(fup, "ssf", "all", "dtd"))
for ds in DS:
    src = fup if ds == "dtd" else main
    auto = acc(fup, "lora", "auto", ds)
    for p in ["early", "even", "late", "all"]:
        add(f"{ds}:lora:auto_vs_{p}", auto, acc(src, "lora", p, ds))
pd.DataFrame(rows).to_csv(os.path.join(FUP, "followup_significance.csv"), index=False)
print("wrote fig6_dtd_placement.png, fig9_fisher_auto.png, followup_significance.csv")
