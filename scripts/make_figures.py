#!/usr/bin/env python3
"""从 results/summary.csv（+ results/runs/）重新生成全部图表(图 1-5)。

用法: python scripts/make_figures.py

所有图都从 117 条原始实验记录(summary.csv)自动聚合得到,不依赖任何手工整理
的中间表,因此完全可复现。图 4 额外读取 results/runs/ 下的逐轮日志。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
RUNS = os.path.join(RES, "runs")
os.makedirs(OUT, exist_ok=True)

# 若 runs/ 不存在但有 runs.zip(为方便上传 GitHub,逐轮日志打包存放),自动解压
import zipfile
if not os.path.isdir(RUNS) and os.path.exists(os.path.join(RES, "runs.zip")):
    with zipfile.ZipFile(os.path.join(RES, "runs.zip")) as _z:
        _z.extractall(RES)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

COL = {"linear": "#9e9e9e", "bitfit": "#4c9f70", "ssf": "#e6a417",
       "lora": "#3b6fb6", "full": "#c0392b"}
LBL = {"linear": "Linear probe", "bitfit": "BitFit", "ssf": "SSF",
       "lora": "LoRA", "full": "Full fine-tune"}
ORDER = ["linear", "bitfit", "ssf", "lora", "full"]
DSS = ["cifar100", "flowers", "pets"]
DL = {"cifar100": "CIFAR-100", "flowers": "Flowers-102", "pets": "Oxford Pets"}

# --------------------------------------------------------------------------- #
#  Load the single source of truth and aggregate over seeds
# --------------------------------------------------------------------------- #
raw = pd.read_csv(os.path.join(RES, "summary.csv"))
raw = raw.drop_duplicates(["method", "placement", "dataset", "seed"], keep="last")


def agg(df):
    """mean / std / n of test_acc plus mean trainable & pct, per group."""
    return df.agg(test_acc=("test_acc", "mean"), test_std=("test_acc", "std"),
                  n=("test_acc", "size"), trainable=("trainable", "mean"),
                  pct=("pct", "mean"))


# method-level table (adapt-all configs), indexed by (method, dataset)
allcfg = (raw[raw["placement"].eq("all")]
          .groupby(["method", "dataset"]).pipe(lambda g: agg(g)).reset_index())


def cell(method, dataset, col="test_acc"):
    r = allcfg[(allcfg.method == method) & (allcfg.dataset == dataset)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# --------------------------------------------------------------------------- #
#  Fig 1 — accuracy vs. parameter cost on CIFAR-100 (scatter, log-x)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.6, 4.4))
xs = [cell(m, "cifar100", "trainable") for m in ORDER]
ys = [cell(m, "cifar100", "test_acc") for m in ORDER]
# scatter only -- deliberately NOT connected with a line, so the figure does not
# suggest a continuously tunable Pareto frontier (see the caption of Fig. 1)
for m, x, y in zip(ORDER, xs, ys):
    ax.scatter(x, y, s=130, color=COL[m], edgecolor="black", lw=0.6, zorder=3)
posd = {"linear": (10, 4, "left"), "bitfit": (-9, -15, "right"), "ssf": (-2, 9, "center"),
        "lora": (10, -4, "left"), "full": (10, 2, "left")}
for m, x, y in zip(ORDER, xs, ys):
    dx, dy, ha = posd[m]
    ax.annotate(LBL[m], (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9.5)
ax.set_xscale("log")
ax.set_xlabel("Trainable parameters (log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs. parameter cost on CIFAR-100\n(ViT-B/16, ImageNet-21k pretrained, 3-seed mean)")
ax.set_ylim(85, 94.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_pareto_cifar100.png"))
plt.close(fig)

# --------------------------------------------------------------------------- #
#  Fig 2 — five methods across three datasets (grouped bars, 3-seed mean±std)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(3); w = 0.16
for j, m in enumerate(ORDER):
    vals = [cell(m, ds, "test_acc") for ds in DSS]
    errs = [cell(m, ds, "test_std") for ds in DSS]
    b = ax.bar(x + (j - 2) * w, vals, w, yerr=errs, capsize=2.5,
               color=COL[m], edgecolor="black", lw=0.4, label=LBL[m])
    for bb, v in zip(b, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.15, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.2, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([DL[d] for d in DSS])
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(84, 101.5)
ax.set_title("Fine-tuning methods across three datasets (3-seed mean ± std)")
ax.legend(ncol=5, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_methods_by_dataset.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
#  Fig 3 — placement ablation on CIFAR-100 (mean ± std, 3 seeds)
# --------------------------------------------------------------------------- #
def ms(method, placement, ds="cifar100"):
    g = raw[(raw.method == method) & (raw.placement == placement) & (raw.dataset == ds)]["test_acc"]
    return g.mean(), (g.std() if len(g) > 1 else 0.0)


places = ["early", "mid", "even", "late"]
xl = ["early\n(0–3)", "mid\n(4–7)", "even\n(0,3,6,9)", "late\n(8–11)"]
x = np.arange(4); w = 0.36
fig, ax = plt.subplots(figsize=(7.4, 4.6))
lm = [ms("lora", p)[0] for p in places]; ls_ = [ms("lora", p)[1] for p in places]
sm = [ms("ssf", p)[0] for p in places]; ss_ = [ms("ssf", p)[1] for p in places]
ax.bar(x - w / 2, lm, w, yerr=ls_, capsize=4, color=COL["lora"], edgecolor="black", lw=0.4,
       label="LoRA (4 blocks, 0.20%)")
ax.bar(x + w / 2, sm, w, yerr=ss_, capsize=4, color=COL["ssf"], edgecolor="black", lw=0.4,
       label="SSF (4 blocks, 0.17%)")
for xi, m in zip(x - w / 2, lm): ax.text(xi, m + 0.05, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
for xi, m in zip(x + w / 2, sm): ax.text(xi, m + 0.05, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
la = ms("lora", "all")[0]; sa = ms("ssf", "all")[0]
ax.axhline(la, color=COL["lora"], ls="--", lw=1.1); ax.axhline(sa, color=COL["ssf"], ls="--", lw=1.1)
ax.text(3.4, la, "LoRA all-12 (0.43%)", color=COL["lora"], fontsize=7.5, va="bottom", ha="right")
ax.text(3.4, sa, "SSF all-12 (0.33%)", color=COL["ssf"], fontsize=7.5, va="top", ha="right")
ax.set_xticks(x); ax.set_xticklabels(xl)
ax.set_xlabel("Which 4 transformer blocks are adapted")
ax.set_ylabel("Test accuracy (%)  (mean ± std, 3 seeds)"); ax.set_ylim(89.5, 93.4)
ax.set_title("Where to spend a fixed PEFT budget (CIFAR-100, 3 seeds)")
ax.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_placement_ablation.png"))
plt.close(fig)

# --------------------------------------------------------------------------- #
#  Fig 4 — LoRA validation curves by placement (CIFAR-100, seed 42)
# --------------------------------------------------------------------------- #
cf = {"all (0–11)": "lora_cifar100_seed42.csv", "early (0–3)": "lora-early_cifar100_seed42.csv",
      "mid (4–7)": "lora-mid_cifar100_seed42.csv", "late (8–11)": "lora-late_cifar100_seed42.csv"}
cc = {"all (0–11)": "#3b6fb6", "early (0–3)": "#4c9f70", "mid (4–7)": "#e6a417", "late (8–11)": "#c0392b"}
if all(os.path.exists(os.path.join(RUNS, f)) for f in cf.values()):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for lab, fn in cf.items():
        c = pd.read_csv(os.path.join(RUNS, fn))
        ax.plot(c["epoch"], c["val_acc"], marker="o", ms=3, lw=1.6, color=cc[lab], label="LoRA " + lab)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("LoRA training curves by placement (CIFAR-100, seed 42)")
    ax.legend(loc="lower right", fontsize=9); ax.set_ylim(78, 94)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_val_curves.png"))
    plt.close(fig)
    fig4 = True
else:
    fig4 = False

# --------------------------------------------------------------------------- #
#  Fig 5 — mechanism check: placement ablation on all three datasets
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Mechanism validation: does the early-layer advantage persist on native high-resolution data?\n"
             "(Error bars = std over 3 seeds;  dashed = all-12 layers reference)", fontsize=12)
ylims = {"cifar100": (89.5, 93.4), "flowers": (98.4, 99.7), "pets": (91.5, 95.5)}
for row, method in enumerate(["lora", "ssf"]):
    for col, ds in enumerate(DSS):
        ax = axes[row][col]
        vals = [ms(method, p, ds)[0] for p in places]
        errs = [ms(method, p, ds)[1] for p in places]
        ax.bar(np.arange(4), vals, 0.62, yerr=errs, capsize=4,
               color=COL[method], edgecolor="black", lw=0.4)
        for xi, v in zip(np.arange(4), vals):
            ax.text(xi, v + (ylims[ds][1] - ylims[ds][0]) * 0.012, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        allv = ms(method, "all", ds)[0]
        ax.axhline(allv, color="#555", ls="--", lw=1.0)
        ax.text(3.45, allv, "all-12", color="#555", fontsize=7.5, va="bottom", ha="right")
        ax.set_xticks(np.arange(4)); ax.set_xticklabels(xl, fontsize=9)
        ax.set_ylim(*ylims[ds])
        ax.set_title(f"{LBL[method]}  —  {DL[ds]}", fontsize=10.5)
        if col == 0:
            ax.set_ylabel("Test accuracy (%) / mean±std, 3 seeds", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig5_mechanism.png"))
plt.close(fig)

print("wrote fig1-3, fig5" + (", fig4" if fig4 else " (fig4 skipped: runs/ not found)"))
