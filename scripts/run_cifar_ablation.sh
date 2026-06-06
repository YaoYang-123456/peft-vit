#!/bin/bash
# 复跑 CIFAR-100 的层位置消融(3 种子)+ even 对照 + 效率基线;结果写入项目内 results/。
# 断点续跑:已完成的按 results/done_*.flag 跳过。后台运行: nohup bash scripts/run_all.sh > train.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1                    # 切到项目根目录
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DATA=./data; OUT=./results
mkdir -p "$DATA" "$OUT"
run () {   # $1=method  $2=epochs  $3=seed
  local m="$1" ep="$2" sd="$3" base pos
  base="${m%%-*}"; if [ "$m" = "$base" ]; then pos="all"; else pos="${m#*-}"; fi
  local flag="$OUT/done_${base}_${pos}_${sd}.flag"
  if [ -f "$flag" ]; then echo "== skip $m seed$sd"; return; fi
  echo "===== $m / cifar100 / seed $sd ($ep ep) ====="
  python train.py --method "$m" --dataset cifar100 --epochs "$ep" --seed "$sd" \
    --num-workers 4 --data-root "$DATA" --out-dir "$OUT" && touch "$flag"
}
for s in 42 43 44; do for b in lora ssf; do for p in all early mid late; do
  if [ "$p" = "all" ]; then run "$b" 20 "$s"; else run "$b-$p" 20 "$s"; fi
done; done; done
for s in 42 43 44; do for b in lora ssf; do run "$b-even" 20 "$s"; done; done
run linear 8 42; run bitfit 20 42; run full 20 42
echo "ALL_DONE  (then: python scripts/make_figures.py)"
