#!/bin/bash
# 复跑 CIFAR-100 的层位置消融(3 种子)+ even 对照 + 效率基线;结果写入项目内 results/。
# 断点续跑:已完成的按 results/summary.csv 是否已有对应行跳过(无需 flag 文件)。后台运行: nohup bash scripts/run_cifar_ablation.sh > train.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1                    # 切到项目根目录
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DATA=./data; OUT=./results
mkdir -p "$DATA" "$OUT"
already_done () {   # $1=method(base)  $2=placement  $3=seed  (dataset 恒为 cifar100)
  local csv="$OUT/summary.csv"
  [ -f "$csv" ] || return 1
  awk -F, -v m="$1" -v p="$2" -v s="$3" \
    'NR>1 && $1==m && $2==p && $3=="cifar100" && $4==s {found=1} END{exit !found}' "$csv"
}
run () {   # $1=method  $2=epochs  $3=seed
  local m="$1" ep="$2" sd="$3" base pos
  base="${m%%-*}"; if [ "$m" = "$base" ]; then pos="all"; else pos="${m#*-}"; fi
  if already_done "$base" "$pos" "$sd"; then echo "== skip $m seed$sd(summary.csv 已有)"; return; fi
  echo "===== $m / cifar100 / seed $sd ($ep ep) ====="
  python train.py --method "$m" --dataset cifar100 --epochs "$ep" --seed "$sd" \
    --num-workers 4 --data-root "$DATA" --out-dir "$OUT"
}
for s in 42 43 44; do for b in lora ssf; do for p in all early mid late; do
  if [ "$p" = "all" ]; then run "$b" 20 "$s"; else run "$b-$p" 20 "$s"; fi
done; done; done
for s in 42 43 44; do for b in lora ssf; do run "$b-even" 20 "$s"; done; done
run linear 20 42; run bitfit 20 42; run full 20 42
echo "ALL_DONE  (then: python scripts/make_figures.py)"
