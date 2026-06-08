# PEFT-ViT — 视觉 Transformer 高效微调方法对比与层位置分析

在 ImageNet-21k 预训练的 **ViT-B/16** 上,统一协议对比五种微调方法(线性探测 / 全量微调 / BitFit / LoRA / SSF),
并对 LoRA、SSF 做 **3 个随机种子的层位置消融**(固定预算下适配早/中/晚/均匀分布的 Transformer Block),
据此提出 **“深度感知 PEFT(Depth-aware PEFT)”** 改进,并对比五种方法的显存与时间开销。

## 主要结论
- PEFT 仅用 **<0.5% 参数**即可恢复全量微调约 **99%** 的精度。
- **小数据上 PEFT 反超全量微调**(**3 种子确认**):Flowers-102 上 LoRA 99.31±0.15% vs Full 95.78±**0.88**%(差 3.5 pp);Pets 上 93.97±0.37% vs 92.36±0.77%(差 1.6 pp)。Full FT 方差是 PEFT 的 2–10 倍,说明小数据上全量微调不仅精度低,结果也更不稳定。
- **层位置消融**:CIFAR-100(低分辨率)上,只适配晚层明显最差(1.5–2.5 pp,配对 t 检验 p<0.05);早/中/均匀放法以约 **⅓ 适配参数**逼近全部 12 层。
- **机制验证**:在高分辨率 Flowers/Pets 上重做消融,层位置影响**消失**(<0.3 pp,无显著差异)。这证实了机制预测:层位置效应来自低分辨率数据与预训练分布的偏移,当数据与 ImageNet 分布接近时层位置选择几乎自由。
- LoRA **显存与速度最均衡**;SSF 在本实现下显存反而高于全量微调。

## 实验结果
**五种方法测试精度(%)(3 种子均值±标准差)**

| 方法 | CIFAR-100 | Flowers-102 | Pets |
|---|---|---|---|
| Linear probe | 86.35±0.46 | 98.85±0.26 | 92.10±0.57 |
| BitFit | 92.65±0.29 | 99.12±0.16 | 93.74±0.38 |
| SSF | 92.78±0.17 | 99.22±0.08 | 93.79±0.62 |
| LoRA | 92.56±0.09 | 99.31±0.15 | 93.97±0.37 |
| Full fine-tune | 93.35±0.14 | **95.78±0.88** | **92.36±0.77** |

**CIFAR-100 层位置消融(3 种子)**:LoRA early/mid/even ≈92.2–92.4% vs late 90.91%;SSF early/mid/even ≈92.2–92.6% vs late 90.40%。Flowers/Pets 上所有放法精度相近(<0.3 pp)——机制得到验证。

## 代码与工具来源声明
本项目基于 PyTorch、torchvision 与 timm 搭建训练框架;BitFit、LoRA、SSF 三种微调方法均根据各自原文献(见报告参考文献 [4][5][6])**独立重实现**(实现见 `src/methods.py`),**未使用 loralib、peft 等现成 PEFT 库,也未复制任何公开代码或微调仓库**;训练流程与实验设计均为原创,三个数据集均通过 torchvision 标准接口加载。实验在作者自有代码仓库上运行,逐轮训练日志见 `results/runs/`。

## 实验环境
- Python ≥ 3.10,单张 GPU(本项目在 RTX 4090D 上验证;PyTorch 2.x / CUDA 11.8+)。
- 安装依赖:`pip install -r requirements.txt`(含 `torch / torchvision / timm / pandas / numpy / matplotlib / scipy`)。实验在 AutoDL 上以 Python 3.12 + PyTorch 2.x + timm(最新版)完成;如需完全锁定版本,可在该环境运行 `pip freeze > requirements-lock.txt` 一并提交。
- 查看版本:`python -c "import torch,timm;print('torch',torch.__version__,'timm',timm.__version__)"`。

## 数据集准备
CIFAR-100、Flowers-102、Oxford-IIIT Pets 均由 `torchvision` 自动下载到 `--data-root`(默认 `./data`)。
验证集为从训练集划出的 10%(Flowers-102 用其官方 train/val/test 划分),测试集仅在训练结束时评测一次。

一次性下载并自检(打印 `ALL DATA READY` 即就绪):
```bash
python -c "
import torchvision as tv
r='./data'
tv.datasets.CIFAR100(r, train=True, download=True)
tv.datasets.Flowers102(r, split='train', download=True)
tv.datasets.OxfordIIITPet(r, split='trainval', download=True)
print('ALL DATA READY')
"
```
> 国内下载慢时:CIFAR-100 可手动把 `cifar-100-python.tar.gz` 放入 `--data-root` 跳过下载;
> 预训练权重可设 `export HF_ENDPOINT=https://hf-mirror.com` 走镜像。

## 运行方式
**单个配置**(`method` 后加后缀切换层位置/秩;不加后缀 = 适配全部 12 层):
```bash
python train.py --method <linear|full|bitfit|lora|ssf> --dataset <cifar100|flowers|pets> \
                --epochs N --seed 42 --data-root ./data --out-dir ./results
python train.py --method lora-early --dataset cifar100 --epochs 20   # 早段 0-3
python train.py --method ssf-mid    --dataset cifar100 --epochs 20   # 中段 4-7
python train.py --method lora-even  --dataset cifar100 --epochs 20   # 均匀 0,3,6,9
python train.py --method lora-r12-early --dataset cifar100           # 早段 + LoRA 秩=12
```
可用位置后缀:`early / mid / late / even / earlymid`;LoRA 秩用 `r<N>`。每个 run 把一行结果追加到
`results/summary.csv`,逐轮日志写入 `results/runs/`。

**快速冒烟测试**(1–3 分钟确认环境与代码跑通;先装依赖 `pip install -r requirements.txt`):
```bash
bash scripts/smoke_test.sh
```

**复跑全部实验 → 再生成表与图**(已完成的 run 自动跳过,非从 checkpoint 续训;后台:`nohup ... > train.log 2>&1 &`):
```bash
bash scripts/run_all_full.sh        # 全部 117 个 run(也可用 run_cifar_ablation.sh 只跑 CIFAR 消融)
python scripts/summarize_results.py # 从 summary.csv 再生成表 2/3/4/5 + p 值表
python scripts/make_figures.py      # 从 summary.csv(+runs/)再生成图 1–5
```

**结果文件说明**
- `results/summary.csv` — 全部实验结果(117 行;CIFAR-100 + Flowers-102 + Pets,3 种子)
- `results/main_comparison.csv` — 表 2(五方法 × 三数据集,3 种子均值/标准差/n)
- `results/placement_ablation.csv` — 表 3/4(LoRA/SSF 层位置,各数据集 mean±std)
- `results/significance.csv` — 配对 t 检验 p 值(§3.3/§3.4 引用)
- `results/efficiency.csv` — 表 5(峰值显存=各种子一致值;每轮时间=seed42 代表性运行)
- `results/runs.zip` — 117 个 run 的逐轮日志(为便于上传 GitHub 已打包;运行 `make_figures.py` / `summarize_results.py` 会自动解压,或手动 `cd results && unzip runs.zip`)。其中 `ssf_cifar100_seed42/43/44.csv` 是中途续跑的日志、自第 3/5 轮起记录,但末轮的 best_val/test 与 `summary.csv` 完全一致,且 fig4 使用 LoRA 曲线,故不影响任何表与图
- `results/figures/` — 图 1–7
- `results/eqbudget/` — §3.6 等预算重分配实验(4 块 × r24,CIFAR-100;summary.csv + runs.zip)
- `results/baseline_sweep/` — §3.6 全量微调在 Flowers 上的学习率体检(lr=3e-5 / 5e-5)
- `results/vit_small/` — §3.7 第二主干 ViT-Small/16 的 LoRA 层位置消融(CIFAR-100 + Flowers,3 种子;summary.csv + runs.zip)

## 代码结构
```
train.py                     训练入口(一个 method × dataset 配置)
src/backbone.py              ViT-B/16 (IN-21k) 主干
src/data.py                  数据集、数据增广与 DataLoader
src/methods.py               五种微调方法 + 层位置/秩 (configure_method)
src/engine.py                训练/评估循环(AMP、warmup + 余弦退火)
src/utils.py                 随机种子、参数统计、日志
scripts/run_all_full.sh      完整复跑全部 117 个 run(3 数据集 × 3 种子;已完成的 run 自动跳过)
scripts/run_cifar_ablation.sh  仅复跑 CIFAR-100 层位置消融(较快)
scripts/summarize_results.py  从 summary.csv 再生成表 2/3/4/5 及配对 t 检验 p 值
scripts/extra_experiments.py  再现 §3.6/§3.7 追加实验(等预算重分配、基线体检、第二主干 ViT-S)+ 生成图 6/7
scripts/smoke_test.sh        1–3 分钟快速冒烟测试(先装依赖,确认代码端到端跑通)
scripts/make_figures.py      从 summary.csv(+runs/)重新生成图 1–5
results/                     summary.csv / main_comparison.csv / placement_ablation.csv
                             / significance.csv / efficiency.csv / runs/ / figures/
```
