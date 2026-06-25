#!/usr/bin/env python3
"""一键从代码再生成论文最终使用的全部表格(表 2–表 7),便于"代码→论文表格"完整可复现。

数据来源(两份 summary 合并;均按 method/placement/dataset/seed 去重保留最新):
  - results/summary.csv            主体实验:CIFAR-100 / Flowers-102 / Oxford Pets
  - results/followup/summary.csv   追加实验:DTD,以及 Fisher 自动选层(lora-auto)
  - results/followup/fisher_*.json  各数据集 × 种子 Fisher 选中的 Block
  - results/runs/ (或 runs.zip)     每轮日志,用于表 6 的"每轮时间"

生成的表:
  表 2  五方法 × 四数据集 测试精度(3 种子 mean±std)+ 可训练参数(以 CIFAR-100 计)
  表 3  CIFAR-100 上 LoRA/SSF 层位置消融
  表 4  Flowers-102 与 Oxford Pets 上层位置消融
  表 5  DTD 上层位置消融
  表 6  CIFAR-100 训练开销(峰值显存 + 每轮时间)
  表 7  Fisher 自动选层 vs 人工放法

用法: python scripts/make_report_tables.py
  默认打印 Markdown 表格;同时把每张表写入 results/report_tables/*.csv。
  (可在实验未跑完时运行,缺失的格子会显示 n<3 或 -- ,不影响已完成部分。)
"""
import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RUNS = os.path.join(RES, "runs")
FUP = os.path.join(RES, "followup")
OUTDIR = os.path.join(RES, "report_tables")
KEY = ["method", "placement", "dataset", "seed"]

# 表 6 每轮时间需要逐轮日志;若只打包了 runs.zip 则自动解压(与 summarize_results.py 一致)。
if not os.path.isdir(RUNS) and os.path.exists(os.path.join(RES, "runs.zip")):
    with zipfile.ZipFile(os.path.join(RES, "runs.zip")) as _z:
        _z.extractall(RES)

MNAME = {"linear": "线性探测", "bitfit": "BitFit", "ssf": "SSF",
         "lora": "LoRA", "full": "全量微调"}
PLACE_BLOCKS = {"all": "0–11", "early": "0–3", "mid": "4–7",
                "even": "0,3,6,9", "late": "8–11"}


def load(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=KEY + ["trainable", "total", "pct",
                                           "test_acc", "peak_mem_mb"])
    return pd.read_csv(path).drop_duplicates(KEY, keep="last")


MAIN = load(os.path.join(RES, "summary.csv"))
FOLLOW = load(os.path.join(FUP, "summary.csv"))


def src_for(ds, placement="all"):
    """DTD 在 followup;Fisher 自动选层(placement=auto)的 run 也全部写在 followup;
    其余(cifar100/flowers/pets 的人工放法)在主结果。"""
    if placement == "auto" or ds == "dtd":
        return FOLLOW
    return MAIN


def rows(df, method, placement, ds):
    g = df[(df.method == method) & (df.placement == placement) &
           (df.dataset == ds)].sort_values("seed")
    return g


def acc_ms(method, placement, ds):
    """返回 (mean, std, n);自动选择该 (placement, 数据集) 对应的 summary 源。"""
    v = rows(src_for(ds, placement), method, placement, ds)["test_acc"].values
    if len(v) == 0:
        return None, None, 0
    mean = round(float(np.mean(v)), 2)
    std = round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else float("nan")
    return mean, std, len(v)


def cell(method, placement, ds):
    m, s, n = acc_ms(method, placement, ds)
    if n == 0:
        return "--"
    if n < 2 or s != s:                       # n==1 或 std 为 NaN
        return f"{m:.2f} (n={n})"
    return f"{m:.2f} ± {s:.2f}"


def trainable_of(method, placement, ds):
    g = rows(src_for(ds, placement), method, placement, ds)
    return int(g["trainable"].iloc[0]) if len(g) else None


def pct_str(method, ds="cifar100"):
    """以可训练参数/总参数直接计算占比,避免 CSV 中 pct 预舍入造成的二次舍入误差。"""
    g = rows(src_for(ds), method, "all", ds)
    if not len(g):
        return "--"
    tr = float(g["trainable"].iloc[0])
    tot = float(g["total"].iloc[0])
    p = tr / tot * 100.0
    return "100%" if round(p) >= 100 else f"{p:.3f}%"


# 收集 Markdown 文本 + 落盘 CSV
os.makedirs(OUTDIR, exist_ok=True)
_buf = []


def emit(line=""):
    _buf.append(line)
    print(line)


def save_csv(name, df):
    df.to_csv(os.path.join(OUTDIR, name), index=False)


# --------------------------------------------------------------------------- #
#  表 2  五方法 × 四数据集
# --------------------------------------------------------------------------- #
emit("## 表 2  五种微调方法的测试精度(%,3 种子均值±标准差;参数以 CIFAR-100 计)")
emit()
emit("| 方法 | 可训练参数 | 占比 | CIFAR-100 | Flowers-102 | Oxford Pets | DTD |")
emit("| --- | --- | --- | --- | --- | --- | --- |")
t2 = []
for m in ["linear", "bitfit", "ssf", "lora", "full"]:
    tr = trainable_of(m, "all", "cifar100")
    tr_s = f"{tr:,}" if tr is not None else "--"
    c = cell(m, "all", "cifar100")
    fl = cell(m, "all", "flowers")
    pe = cell(m, "all", "pets")
    dt = cell(m, "all", "dtd")
    emit(f"| {MNAME[m]} | {tr_s} | {pct_str(m)} | {c} | {fl} | {pe} | {dt} |")
    r = {"method": m, "trainable": tr, "pct": pct_str(m)}
    for ds, lab in [("cifar100", "cifar100"), ("flowers", "flowers"),
                    ("pets", "pets"), ("dtd", "dtd")]:
        mn, sd, n = acc_ms(m, "all", ds)
        r[f"{lab}_mean"], r[f"{lab}_std"], r[f"{lab}_n"] = mn, sd, n
    t2.append(r)
save_csv("table2_main_comparison.csv", pd.DataFrame(t2))
emit()


def placement_block(method, p, ds):
    tr = trainable_of(method, p, ds)
    return PLACE_BLOCKS.get(p, "?"), (f"{tr:,}" if tr is not None else "--")


# --------------------------------------------------------------------------- #
#  表 3  CIFAR-100 层位置消融
# --------------------------------------------------------------------------- #
emit("## 表 3  CIFAR-100 上 LoRA/SSF 的层位置消融(固定预算,仅位置不同;3 种子均值±标准差)")
emit()
emit("| 配置 | 适配 Block | 可训练参数 | 测试精度(%) |")
emit("| --- | --- | --- | --- |")
t3 = []
for method in ["lora", "ssf"]:
    for p in ["all", "early", "mid", "even", "late"]:
        blk, tr_s = placement_block(method, p, "cifar100")
        c = cell(method, p, "cifar100")
        tag = "全部(对照)" if p == "all" else {"early": "早段", "mid": "中段",
                                              "even": "均匀", "late": "晚段"}[p]
        emit(f"| {method.upper()} — {tag} | {blk} | {tr_s} | {c} |")
        mn, sd, n = acc_ms(method, p, "cifar100")
        t3.append(dict(method=method, placement=p, blocks=blk,
                       trainable=trainable_of(method, p, "cifar100"),
                       test_mean=mn, test_std=sd, n=n))
save_csv("table3_cifar_placement.csv", pd.DataFrame(t3))
emit()

# --------------------------------------------------------------------------- #
#  表 4  Flowers / Pets 层位置消融
# --------------------------------------------------------------------------- #
emit("## 表 4  Flowers-102 与 Oxford Pets 上的层位置消融(3 种子均值±标准差)")
emit()
emit("| 配置 | 适配 Block | Flowers-102 (%) | Oxford Pets (%) |")
emit("| --- | --- | --- | --- |")
t4 = []
for method in ["lora", "ssf"]:
    for p in ["all", "early", "mid", "even", "late"]:
        blk = PLACE_BLOCKS[p]
        fl = cell(method, p, "flowers")
        pe = cell(method, p, "pets")
        tag = "全部" if p == "all" else {"early": "早", "mid": "中",
                                        "even": "均匀", "late": "晚"}[p]
        emit(f"| {method.upper()} — {tag} | {blk} | {fl} | {pe} |")
        fm, fs, fn = acc_ms(method, p, "flowers")
        pm, ps, pn = acc_ms(method, p, "pets")
        t4.append(dict(method=method, placement=p, blocks=blk,
                       flowers_mean=fm, flowers_std=fs,
                       pets_mean=pm, pets_std=ps))
save_csv("table4_flowers_pets_placement.csv", pd.DataFrame(t4))
emit()

# --------------------------------------------------------------------------- #
#  表 5  DTD 层位置消融
# --------------------------------------------------------------------------- #
emit("## 表 5  DTD 上的层位置消融(3 种子均值±标准差;固定预算每放法适配 4 块)")
emit()
emit("| 配置 | 适配 Block | LoRA (%) | SSF (%) |")
emit("| --- | --- | --- | --- |")
t5 = []
for p in ["all", "early", "mid", "even", "late"]:
    blk = PLACE_BLOCKS[p]
    lo = cell("lora", p, "dtd")
    ss = cell("ssf", p, "dtd")
    tag = "全部(对照)" if p == "all" else {"early": "早段", "mid": "中段",
                                          "even": "均匀", "late": "晚段"}[p]
    emit(f"| {tag} | {blk} | {lo} | {ss} |")
    lm, ls, ln = acc_ms("lora", p, "dtd")
    sm, ss_, sn = acc_ms("ssf", p, "dtd")
    t5.append(dict(placement=p, blocks=blk, lora_mean=lm, lora_std=ls,
                   ssf_mean=sm, ssf_std=ss_))
save_csv("table5_dtd_placement.csv", pd.DataFrame(t5))
emit()

# --------------------------------------------------------------------------- #
#  表 6  CIFAR-100 训练开销
# --------------------------------------------------------------------------- #
emit("## 表 6  五种方法在 CIFAR-100 上的训练开销(均适配全部层;峰值显存各种子一致,每轮时间为 seed42)")
emit()
emit("| 方法 | 可训练参数 | 占比 | 峰值显存(MB) | 每轮时间(s) |")
emit("| --- | --- | --- | --- | --- |")
t6 = []
for m in ["linear", "bitfit", "lora", "ssf", "full"]:
    g = rows(MAIN, m, "all", "cifar100")
    if not len(g):
        emit(f"| {MNAME[m]} | -- | -- | -- | -- |")
        continue
    tr = int(g["trainable"].iloc[0])
    mem = int(round(float(g["peak_mem_mb"].iloc[0])))
    f = os.path.join(RUNS, f"{m}_cifar100_seed42.csv")
    spe = round(float(pd.read_csv(f)["time_s"].mean()), 1) if os.path.exists(f) else None
    spe_s = f"{spe:.1f}" if spe is not None else "--"
    emit(f"| {MNAME[m]} | {tr:,} | {pct_str(m)} | {mem:,} | {spe_s} |")
    t6.append(dict(method=m, trainable=tr, pct=pct_str(m),
                   peak_mem_MB=mem, sec_per_epoch=spe))
save_csv("table6_efficiency.csv", pd.DataFrame(t6))
emit()

# --------------------------------------------------------------------------- #
#  表 7  Fisher 自动选层 vs 人工放法
# --------------------------------------------------------------------------- #
emit("## 表 7  Fisher 自动选层与人工放法的对比(LoRA, r=8, k=4;3 种子均值±标准差)")
emit()
emit("| 数据集 | 选中的 Block(3 种子) | auto | 早段 | 均匀 | 晚段 | 全部 12 层 |")
emit("| --- | --- | --- | --- | --- | --- | --- |")
DSNAME = {"cifar100": "CIFAR-100", "flowers": "Flowers-102", "dtd": "DTD"}
t7 = []
for ds in ["cifar100", "flowers", "dtd"]:
    sel = {}
    for f in sorted(glob.glob(os.path.join(FUP, f"fisher_{ds}_seed*.json"))):
        j = json.load(open(f))
        sel[j["seed"]] = j["selected"]
    if sel:
        uniq = {tuple(b) for b in sel.values()}
        if len(uniq) == 1:
            sel_s = "{" + ",".join(map(str, sorted(next(iter(uniq))))) + "}(三种子一致)"
        else:
            sel_s = "; ".join("seed{}:{{{}}}".format(s, ",".join(map(str, b)))
                              for s, b in sorted(sel.items()))
    else:
        sel_s = "--"
    auto = cell("lora", "auto", ds)
    early = cell("lora", "early", ds)
    even = cell("lora", "even", ds)
    late = cell("lora", "late", ds)
    allc = cell("lora", "all", ds)
    emit(f"| {DSNAME[ds]} | {sel_s} | {auto} | {early} | {even} | {late} | {allc} |")
    am, asd, an = acc_ms("lora", "auto", ds)
    em, esd, en = acc_ms("lora", "early", ds)
    vm, vsd, vn = acc_ms("lora", "even", ds)
    lm, lsd, ln = acc_ms("lora", "late", ds)
    alm, alsd, aln = acc_ms("lora", "all", ds)
    t7.append(dict(dataset=ds, selected=sel_s, n=an,
                   auto_mean=am, auto_std=asd,
                   early_mean=em, early_std=esd,
                   even_mean=vm, even_std=vsd,
                   late_mean=lm, late_std=lsd,
                   all_mean=alm, all_std=alsd))
save_csv("table7_fisher_auto.csv", pd.DataFrame(t7))
emit()

emit(f"(已将 7 张表写入 {os.path.relpath(OUTDIR, ROOT)}/ 下的 CSV;以上 Markdown 可直接对照论文表 2–表 7。)")
