# PEFT-ViT — 视觉 Transformer 高效微调方法对比与层位置分析

在 ImageNet-21k 预训练的 **ViT-B/16** 上,统一协议对比五种微调方法(线性探测 / 全量微调 / BitFit / LoRA / SSF),
并对 LoRA、SSF 做 **3 个随机种子的层位置消融**(固定预算下适配早/中/晚/均匀分布的 Transformer Block)。
在覆盖三类分布偏移的 **四个数据集**(CIFAR-100 / Flowers-102 / Pets / DTD)上完成机制的**双向验证**,
据此把 **“深度感知 PEFT(Depth-aware PEFT)”** 明确为“适配深度匹配偏移层级”的经验原则,
并给出基于经验 **Fisher 信息的一次性自动选层**及其适用范围的实证;同时对比五种方法的显存与时间开销。

**代码仓库**:<https://github.com/YaoYang-123456/peft-vit>(本压缩包与仓库内容一致)

## 主要结论
- PEFT 仅用 **<0.5% 参数**即可恢复全量微调约 **99%** 的精度;小数据上可匹敌经学习率调优的全量微调(默认 lr 下的“反超”主要来自未调参基线,见报告 §3.7)。
- **层位置取决于偏移类型(核心发现)**:
  - 低级统计偏移(CIFAR-100,32×32 放大):只适配**晚层显著最差**(低 1.5–2.5 pp,配对 t 检验 p<0.05),早/中/均匀放法以约 ⅓ 适配器参数逼近全 12 层;
  - 低级偏移较小(高分辨率 Flowers/Pets):层位置影响**大幅减弱**(Flowers 上各放法相差约 0.3 pp;Pets 上有小幅波动但无一致方向),不再出现 CIFAR 那种晚段稳定最差的模式;
  - **语义偏移(DTD,纹理属性):模式对 LoRA 完全反转**——晚段 79.18±0.41 显著高于早段 77.09±0.37(p=0.009),并以 ⅓ 适配器参数**追平全 12 层** 79.06±0.03(p=0.667);SSF 各放法持平。
- **Fisher 自动选层(LoRA, r=8, k=4;单次选层约 2 分钟)**:低级偏移下**成功**(CIFAR-100 三种子稳定选 {0,1,3,4},91.97±0.21 ≈ 人工最优、显著高于晚段);无偏移时**无害**(Flowers 99.26±0.18,选层不稳定恰印证位置无关);语义偏移下**失效**(DTD 选中段 77.73±0.25,显著低于最优晚段 p=0.012)——DTD 晚层梯度份额全网最低却收益最大,说明一阶梯度准则只在低级偏移下可靠。
- PEFT 方法之间总体差异很小,但 DTD 上 **LoRA 显著优于 SSF**(79.06 vs 77.94,p=0.047):语义重组需要子空间级更新能力。
- LoRA **显存与速度最均衡**;SSF 在本实现下显存反而高于全量微调(实现相关)。

## 实验结果
**五种方法测试精度(%)(3 种子均值±标准差;加粗为每列最高)**

| 方法 | CIFAR-100 | Flowers-102 | Pets | DTD |
|---|---|---|---|---|
| Linear probe | 87.01±0.24 | 98.85±0.26 | 92.10±0.57 | 77.04±0.29 |
| BitFit | 92.65±0.29 | 99.12±0.16 | 93.74±0.38 | 77.48±0.11 |
| SSF | 92.78±0.17 | 99.22±0.08 | 93.79±0.62 | 77.94±0.46 |
| LoRA | 92.56±0.09 | **99.31±0.15** | **93.97±0.37** | **79.06±0.03** |
| Full fine-tune | **93.35±0.14** | 95.78±0.88 | 92.36±0.77 | 76.26±1.26 |

(统一训练轮数与调度:CIFAR-100 20 轮、Flowers/Pets/DTD 30 轮,前 10 轮线性预热 + 余弦退火。)

**层位置消融(LoRA,3 种子)**:CIFAR-100 上 early/mid/even ≈92.1–92.4% vs **late 90.91%**(SSF 同模式,late 90.40%);Flowers 各放法相近(约 0.3 pp)、Pets 小幅波动但无一致方向;**DTD 上反转:late 79.18% ≈ all-12 79.06% > early 77.09%**。

**Fisher 自动选层 vs 人工放法(LoRA)**:CIFAR-100 auto 91.97(选{0,1,3,4})、Flowers auto 99.26、DTD auto 77.73(选中段,< late 79.18,p=0.012)。

## 代码与工具来源声明
本项目基于 PyTorch、torchvision 与 timm 搭建训练框架;BitFit、LoRA、SSF 三种微调方法均根据各自原文献(见报告参考文献 [4][5][6])**独立重实现**(实现见 `src/methods.py`),**未使用 loralib、peft 等现成 PEFT 库,也未复制任何公开代码或微调仓库**;Fisher 自动选层准则与脚本(`scripts/fisher_select.py`)亦为原创实现。训练流程与实验设计均为原创,四个数据集均通过 torchvision 标准接口加载。逐轮训练日志见 `results/runs/` 与 `results/followup/runs/`。

## 实验环境
- Python ≥ 3.10,单张 GPU(本项目在 AutoDL 云平台、单张 RTX 5090D (32GB) 上完成;Python 3.12 / PyTorch 2.8 / CUDA 12.8,全程 fp16 混合精度)。
- 安装依赖:`pip install -r requirements.txt`(含 `torch / torchvision / timm / pandas / numpy / matplotlib / scipy`)。如需完全锁定版本,可在运行环境执行 `pip freeze > requirements-lock.txt` 一并提交。
- 查看版本:`python -c "import torch,timm;print('torch',torch.__version__,'timm',timm.__version__)"`。

## 数据集准备
CIFAR-100、Flowers-102、Oxford-IIIT Pets、DTD 均由 `torchvision` 自动下载到 `--data-root`(默认 `./data`)。
验证集为从训练集划出的 10%(Flowers-102 与 DTD 用各自官方 train/val/test 划分),测试集仅在训练结束时评测一次。

一次性下载并自检(打印 `ALL DATA READY` 即就绪):
```bash
python -c "
import torchvision as tv
r='./data'
tv.datasets.CIFAR100(r, train=True, download=True)
tv.datasets.Flowers102(r, split='train', download=True)
tv.datasets.OxfordIIITPet(r, split='trainval', download=True)
tv.datasets.DTD(r, split='train', download=True)
print('ALL DATA READY')
"
```
> 国内下载慢时:CIFAR-100 可手动把 `cifar-100-python.tar.gz` 放入 `--data-root`;DTD 可手动把 `dtd-r1.0.1.tar.gz` 放入 `data/dtd/`;
> 预训练权重可设 `export HF_ENDPOINT=https://hf-mirror.com` 走镜像。

## 运行方式
**单个配置**(`method` 后加后缀切换层位置/秩;不加后缀 = 适配全部 12 层):
```bash
python train.py --method <linear|full|bitfit|lora|ssf> --dataset <cifar100|flowers|pets|dtd> \
                --epochs N --seed 42 --data-root ./data --out-dir ./results
python train.py --method lora-early --dataset cifar100 --epochs 20   # 早段 0-3
python train.py --method lora-even  --dataset cifar100 --epochs 20   # 均匀 0,3,6,9
python train.py --method lora-r12-early --dataset cifar100           # 早段 + LoRA 秩=12
python train.py --method lora --blocks 2,5,8,11 --placement-label custom --dataset dtd --epochs 30  # 显式指定任意层
```
可用位置后缀:`early / mid / late / even / earlymid`;LoRA 秩用 `r<N>`;`--blocks` 可显式给出任意 Block 列表(自动选层即用此接口)。
每个 run 把一行结果追加到 `results/summary.csv`(追加实验写 `results/followup/summary.csv`),逐轮日志写入对应 `runs/`。

**快速冒烟测试**(1–3 分钟):`bash scripts/smoke_test.sh`

**复跑主体实验(117 个 run)→ 表与图**(已完成的 run 自动跳过):
```bash
bash scripts/run_all_full.sh        # 或 run_cifar_ablation.sh 只跑 CIFAR 消融
python scripts/summarize_results.py # 再生成表 2/3/4/6 + p 值表
python scripts/make_figures.py      # 再生成图 1–5
python scripts/extra_experiments.py # §3.7/§3.8 等预算、基线体检、ViT-S + 图 7/8
```

**复跑追加实验(48 个 run:DTD 全套 + Fisher 自动选层)→ 表与图**:
```bash
bash scripts/run_followup.sh              # 39 个 DTD run + 9 次选层 + 9 个 auto run(断点续跑)
python scripts/fisher_select.py --dataset dtd --seed 42 --k 4   # 也可单独跑一次选层(~2 分钟)
python scripts/summarize_followup.py      # 表 A1/A2/B(DTD 五方法、DTD 层位置、auto 对比)+ 配对 t 检验
python scripts/make_figures_followup.py   # 图 6/9 + results/followup/followup_significance.csv
```

**一键复现论文最终表格(表 2–表 7)**(合并主体 + 追加两份结果,直接打印 Markdown 并写入 `results/report_tables/`):
```bash
python scripts/make_report_tables.py
```

**结果文件说明**
- `results/summary.csv` — 主体实验(117 行;CIFAR-100 + Flowers + Pets,3 种子)
- `results/followup/summary.csv` — 追加实验(48 行;DTD 五方法与层位置 + 三数据集 Fisher 自动选层)
- `results/followup/fisher_<dataset>_seed<S>.json` — 9 次选层的逐 Block Fisher 份额与选中层
- `results/followup/followup_significance.csv` — 追加实验配对 t 检验汇总(报告 §3.5/§3.9 引用)
- `results/followup/followup.log` — 追加实验完整运行日志(AutoDL,RTX 5090D)
- `results/main_comparison.csv` / `placement_ablation.csv` / `significance.csv` / `efficiency.csv` — 表 2/3/4/6 与 p 值
- `results/runs.zip`、`results/followup/runs/` — 逐轮训练日志(主体日志为便于上传已打包,脚本会自动解压;其中 `ssf_cifar100_seed42/43/44.csv` 为中途续跑日志、自第 3/5 轮起记录,末轮 best_val/test 与 summary.csv 一致,不影响任何表与图)
- `results/figures/` — 图 1–9(fig6=DTD 层位置,fig7=等预算,fig8=ViT-S,fig9=Fisher 自动选层;v4 起按报告出现顺序重命名,旧 fig6/fig7 即现 fig7/fig8)
- `results/eqbudget/`、`results/baseline_sweep/`、`results/vit_small/` — §3.7/§3.8 追加实验原始数据

## 代码结构
```
train.py                        训练入口(method × dataset;支持 --blocks 显式选层)
src/backbone.py                 ViT-B/16 (IN-21k) 主干
src/data.py                     四个数据集、增广与 DataLoader
src/methods.py                  五种微调方法 + 层位置/秩/任意 block_ids (configure_method)
src/engine.py                   训练/评估循环(AMP、warmup + 余弦退火)
src/utils.py                    随机种子、参数统计、日志
scripts/run_all_full.sh         复跑主体 117 个 run(已完成自动跳过)
scripts/run_followup.sh         复跑追加 48 个 run(DTD + 自动选层;断点续跑)
scripts/fisher_select.py        Fisher 自动选层(头部预热 300 步 → 累积 200 batch 经验 Fisher → 取 top-k)
scripts/run_cifar_ablation.sh   仅复跑 CIFAR-100 层位置消融(较快)
scripts/summarize_results.py    主体实验:表 2/3/4/6 + 配对 t 检验
scripts/summarize_followup.py   追加实验:表 A1/A2/B + 配对 t 检验
scripts/make_report_tables.py   一键合并两份结果再生成论文表 2–表 7 → results/report_tables/
scripts/make_figures.py         图 1–5
scripts/extra_experiments.py    §3.7/§3.8 等预算、基线体检、ViT-S + 图 7/8
scripts/make_figures_followup.py 图 6/9 + followup_significance.csv
scripts/smoke_test.sh           1–3 分钟快速冒烟测试
results/                        主体与追加实验的全部 CSV / 日志 / 图
```
