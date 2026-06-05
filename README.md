# PEFT-ViT — 视觉 Transformer 高效微调方法对比

在 ImageNet-21k 预训练的 **ViT-B/16** 上,统一协议对比五种微调方法
(线性探测 / 全量微调 / BitFit / LoRA / SSF),并对 LoRA、SSF 做**层位置消融**
(固定预算下适配早/中/晚不同深度的 Transformer Block)。

## 实验环境
- Python ≥ 3.10,PyTorch ≥ 2.1,`timm` ≥ 1.0,`torchvision`,`pandas`
- 建议使用 GPU(本项目在 Google Colab 的 NVIDIA T4 / A100 上验证)
- 安装依赖:`pip install -U timm torch torchvision pandas`

## 数据集下载
CIFAR-100、Flowers-102、Oxford-IIIT Pets 均由 `torchvision` 在首次运行时**自动下载**
到 `--data-root`(默认 `./data`),无需手动准备。验证集为从训练集中按固定随机种子划出的 10%
(Flowers-102 使用其官方 train/val/test 划分);测试集仅在训练结束时评测一次。

## 运行方式
单个配置:
```bash
python train.py --method <linear|full|bitfit|lora|ssf> \
                --dataset <cifar100|flowers|pets> \
                --epochs N --data-root ./data --out-dir ./results
```
层位置消融(在方法名后加后缀 `-early` / `-mid` / `-late`,不加后缀即适配全部 12 层):
```bash
python train.py --method lora-early --dataset cifar100 --epochs 20
python train.py --method ssf-mid    --dataset cifar100 --epochs 20
```
每个配置会打印 `best_val / test_acc / 可训练参数`,并把一行结果追加到 `results/summary.csv`。
本项目使用的训练轮数:线性探测 8;其余方法在 Flowers/Pets 上 30、CIFAR-100 上 20。

## 实验结果
**五种方法测试精度(%)**(可训练参数与占比以 CIFAR-100 为例):

| 方法 | 占比 | CIFAR-100 | Flowers-102 | Pets |
|---|---|---|---|---|
| 线性探测 | 0.090% | 86.42 | 98.57 | 92.45 |
| BitFit | 0.209% | 92.59 | 99.17 | 93.70 |
| SSF | 0.325% | 92.72 | 99.12 | 94.06 |
| LoRA | 0.431% | 92.60 | 99.20 | 93.95 |
| 全量微调 | 100% | 93.41 | 96.34 | 92.86 |

**层位置消融(CIFAR-100,固定预算)**:

| 配置 | 适配 Block | 测试精度(%) |
|---|---|---|
| LoRA 全部 / 早 / 中 / 晚 | 0–11 / 0–3 / 4–7 / 8–11 | 92.60 / 92.28 / 92.26 / 91.09 |
| SSF 全部 / 早 / 中 / 晚 | 0–11 / 0–3 / 4–7 / 8–11 | 92.72 / 92.14 / 92.38 / 90.47 |

主要结论:PEFT 用 <0.5% 参数恢复全量微调约 99% 精度、并在小数据上反超;固定预算下早/中层适配显著优于晚层,
且仅用约 1/3 参数即可保持精度。完整分析见课题报告;图见 `results/figures/`,数据见 `results/summary.csv`。

## 代码结构
```
train.py            训练入口(一个 method × dataset 配置)
src/backbone.py     ViT-B/16 (IN-21k) 主干
src/data.py         数据集、数据增广与 DataLoader
src/methods.py      五种微调方法 + 层位置(configure_method)
src/engine.py       训练/评估循环(AMP、warmup+余弦退火)
src/utils.py        随机种子、参数统计、日志等工具
results/            汇总结果 summary.csv 与图 figures/
```
