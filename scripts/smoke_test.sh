#!/bin/bash
# 快速冒烟测试:1–3 分钟确认代码能端到端跑通(非完整实验)。
# 运行前先装依赖:  pip install -r requirements.txt
set -e
cd "$(dirname "$0")/.."

echo "== [1/2] 从已提交结果再生成表与图(无需下载,约数秒)=="
python scripts/summarize_results.py >/dev/null && echo "   summarize_results.py OK(efficiency.csv 等已重建)"
python scripts/make_figures.py     >/dev/null && echo "   make_figures.py OK(fig1–5 已重建)"

echo "== [2/2] LoRA 在 CIFAR-100 子集上训练 1 轮 =="
echo "   (首次运行会下载 CIFAR-100 与预训练权重;之后约 1 分钟)"
python train.py --method lora --dataset cifar100 --epochs 1 --subset-size 256 \
  --batch-size 16 --num-workers 0 --out-dir ./debug_results

echo "✓ 冒烟测试通过:结果聚合 + 训练/评测链路均正常。debug_results/ 可删除。"
