#!/usr/bin/env python3
"""再现报告 §3.6 的两组追加实验(从原始结果聚合,完全可复现):
  实验A 等预算重分配  -> results/eqbudget/  (4 块 × r24,等于全 12 层 × r8 的预算)
  实验B 基线学习率体检 -> results/baseline_sweep/ (full FT 在 Flowers 上 lr=3e-5/5e-5)
并生成 results/figures/fig7_equal_budget.png。
用法: python scripts/extra_experiments.py
"""
import os, zipfile
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES=os.path.join(ROOT,"results"); FIG=os.path.join(RES,"figures"); os.makedirs(FIG,exist_ok=True)
main=pd.read_csv(os.path.join(RES,"summary.csv")).drop_duplicates(["method","placement","dataset","seed"],keep="last")
eq=pd.read_csv(os.path.join(RES,"eqbudget","summary.csv"))

def acc(df,m,p,ds):
    v=df[(df.method==m)&(df.placement==p)&(df.dataset==ds)]["test_acc"]; return v.values

print("="*64); print("实验A 等预算重分配 (CIFAR-100, 全部 0.43% 参数)"); print("="*64)
base=acc(main,"lora","all","cifar100")
print(f"{'全部12层 r8 (基线)':<20} {base.mean():.2f} ± {base.std(ddof=1):.2f}")
for p in ["early","mid","even","late"]:
    r24=acc(eq,"lora",p,"cifar100"); r8=acc(main,"lora",p,"cifar100")
    _,pv=stats.ttest_rel(r24,base)
    print(f"  {p:<6} r24={r24.mean():.2f}±{r24.std(ddof=1):.2f}  (vs 全12层 p={pv:.3f}) | 对照 r8={r8.mean():.2f}")

print("\n"+"="*64); print("实验B 基线学习率体检 (Flowers-102)"); print("="*64)
ff=acc(main,"full","all","flowers")
print(f"full lr=1e-5 (默认): {ff.mean():.2f} ± {ff.std(ddof=1):.2f}")
for lr,fn in [("3e-5","full_lr3e5.csv"),("5e-5","full_lr5e5.csv")]:
    d=pd.read_csv(os.path.join(RES,"baseline_sweep",fn))
    print(f"full lr={lr}: {d.test_acc.mean():.2f} ± {d.test_acc.std(ddof=1):.2f}")
lo=acc(main,"lora","all","flowers")
print(f"LoRA (最优PEFT): {lo.mean():.2f} ± {lo.std(ddof=1):.2f}")

# fig6
def ms(df,p):
    v=acc(df,"lora",p,"cifar100"); return v.mean(), v.std(ddof=1)
places=["early","mid","even","late"]; xl=["early\n(0–3)","mid\n(4–7)","even\n(0,3,6,9)","late\n(8–11)"]
r8=[ms(main,p) for p in places]; r24=[ms(eq,p) for p in places]; allm=ms(main,"all")[0]
x=np.arange(4); w=0.38
fig,ax=plt.subplots(figsize=(7.6,4.6))
ax.bar(x-w/2,[m for m,_ in r8],w,yerr=[s for _,s in r8],capsize=4,color="#9ecae1",edgecolor="black",lw=0.4,label="4 blocks, r=8 (0.20% params)")
ax.bar(x+w/2,[m for m,_ in r24],w,yerr=[s for _,s in r24],capsize=4,color="#3b6fb6",edgecolor="black",lw=0.4,label="4 blocks, r=24 (0.43% params)")
for xi,(m,_) in zip(x-w/2,r8): ax.text(xi,m+0.04,f"{m:.1f}",ha="center",va="bottom",fontsize=7.5)
for xi,(m,_) in zip(x+w/2,r24): ax.text(xi,m+0.04,f"{m:.1f}",ha="center",va="bottom",fontsize=7.5)
ax.axhline(allm,color="#c0392b",ls="--",lw=1.3)
ax.text(3.45,allm+0.03,f"all 12 layers, r=8 (0.43%) = {allm:.2f}",color="#c0392b",fontsize=8,va="bottom",ha="right")
ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_ylim(90.2,93.2)
ax.set_xlabel("Which 4 transformer blocks get LoRA")
ax.set_ylabel("CIFAR-100 test accuracy (%)  (mean±std, 3 seeds)")
ax.set_title("Equal-budget reallocation: coverage beats rank")
ax.legend(loc="lower left",fontsize=8.5); ax.grid(alpha=0.3,axis="y")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig7_equal_budget.png"),dpi=150); plt.close()
print("\n✓ 已生成 results/figures/fig7_equal_budget.png")

# ====== §3.7 第二主干 ViT-Small/16(若存在）======
vs_path = os.path.join(RES, "vit_small", "summary.csv")
if os.path.exists(vs_path):
    print("\n" + "="*64); print("§3.7 第二主干 ViT-Small/16  LoRA 层位置"); print("="*64)
    vs = pd.read_csv(vs_path).drop_duplicates(["placement","dataset","seed"], keep="last")
    def vacc(p, ds): return vs[(vs.placement==p)&(vs.dataset==ds)]["test_acc"].values
    for ds in ["cifar100", "flowers"]:
        print(f"-- {ds} --")
        for p in ["all","early","mid","even","late"]:
            a = vacc(p, ds); print(f"   {p:<6} {a.mean():.2f} ± {a.std(ddof=1):.2f}")
        lt = vacc("late", ds); al = vacc("all", ds)
        print(f"   late vs all: Δ={lt.mean()-al.mean():+.2f}pp")
    # fig7
    places=["all","early","mid","even","late"]; xl=["all\n(0-11)","early\n(0-3)","mid\n(4-7)","even\n(0,3,6,9)","late\n(8-11)"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,ds,ttl,ylim in [(axes[0],"cifar100","CIFAR-100 (low-res 32px)",(86.5,92)),(axes[1],"flowers","Flowers-102 (high-res)",(98.4,99.7))]:
        m=[vacc(p,ds).mean() for p in places]; s=[vacc(p,ds).std(ddof=1) for p in places]
        b=ax.bar(range(5),m,yerr=s,capsize=4,color=["#3b6fb6"]*4+["#c0392b"],edgecolor="black",lw=0.4)
        for bb,mm in zip(b,m): ax.text(bb.get_x()+bb.get_width()/2,mm+(ylim[1]-ylim[0])*0.01,f"{mm:.1f}",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(range(5)); ax.set_xticklabels(xl,fontsize=8.5); ax.set_ylim(*ylim); ax.grid(alpha=0.3,axis="y"); ax.set_axisbelow(True)
        ax.set_title(ttl,fontsize=10.5); ax.set_ylabel("test accuracy (%)" if ds=="cifar100" else "")
    fig.suptitle("ViT-Small/16: late-layer LoRA worst on CIFAR-100, gap vanishes on high-res Flowers-102",fontsize=10.5)
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(os.path.join(FIG,"fig8_vit_small.png"),dpi=150); plt.close()
    print("✓ 已生成 results/figures/fig8_vit_small.png")
