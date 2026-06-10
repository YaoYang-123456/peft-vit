#!/bin/bash
# 完整复现:3 数据集 × 3 种子 × 全部配置 = 117 个 run，写入项目内 results/。
# 已完成的 run 自动跳过(按 results/done_*.flag),非从 checkpoint 续训。
# 后台运行: nohup bash scripts/run_all_full.sh > train.log 2>&1 &
# 仅想快速复现 CIFAR 主消融,用 scripts/run_cifar_ablation.sh。
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DATA=./data; OUT=./results
mkdir -p "$DATA" "$OUT"

run () {   # $1=method  $2=dataset  $3=epochs  $4=seed
  local m="$1" ds="$2" ep="$3" sd="$4" base pos
  base="${m%%-*}"; if [ "$m" = "$base" ]; then pos="all"; else pos="${m#*-}"; fi
  local flag="$OUT/done_${base}_${pos}_${ds}_${sd}.flag"
  if [ -f "$flag" ]; then echo "== skip $m $ds seed$sd"; return; fi
  echo "===== $m / $ds / seed $sd ($ep ep) ====="
  python train.py --method "$m" --dataset "$ds" --epochs "$ep" --seed "$sd" \
    --num-workers 4 --data-root "$DATA" --out-dir "$OUT" && touch "$flag"
}

# 每个数据集的训练轮数(所有方法一致,含线性探测)
declare -A EP_FULL=( [cifar100]=20 [flowers]=30 [pets]=30 )
declare -A EP_LIN=(  [cifar100]=20 [flowers]=30 [pets]=30 )  # 线性探测与其他方法同轮数、同调度(完全统一的协议)

for ds in cifar100 flowers pets; do
  for sd in 42 43 44; do
    # 五种方法整体对比(适配全部层)
    run linear "$ds" "${EP_LIN[$ds]}"  "$sd"
    run bitfit "$ds" "${EP_FULL[$ds]}" "$sd"
    run full   "$ds" "${EP_FULL[$ds]}" "$sd"
    run lora    "$ds" "${EP_FULL[$ds]}" "$sd"
    run ssf     "$ds" "${EP_FULL[$ds]}" "$sd"
    # LoRA/SSF 层位置消融 + even 对照(固定预算,仅位置不同)
    for b in lora ssf; do
      for p in early mid late even; do
        run "$b-$p" "$ds" "${EP_FULL[$ds]}" "$sd"
      done
    done
  done
done

echo "ALL_DONE  ->  接着跑: python scripts/summarize_results.py && python scripts/make_figures.py"
