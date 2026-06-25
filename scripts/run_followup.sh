#!/bin/bash
# 追加实验(对应报告 §3.5 的 DTD 与 §3.9 的 Fisher 自动选层),全部写入 results/followup/,不动主结果。
#   A 部分: 第四个数据集 DTD(纹理,语义偏移大但低级统计接近自然图像)
#           - 五种方法整体对比 × 3 种子            = 15 runs
#           - LoRA/SSF × {early,mid,late,even} × 3 种子 = 24 runs
#   B 部分: Fisher 自动选层("深度感知 PEFT"的自动化)
#           - 每个 (数据集, 种子) 先跑 fisher_select.py 选出 4 个 Block,
#             再用所选 Block 训练 LoRA(r=8,placement 标签 = auto)
#           - cifar100 / flowers / dtd × 3 种子      = 9 selections + 9 runs
# 预计总耗时:RTX 5090D 上约 3.5–4.5 小时。
# 断点续跑:已完成的 run 会按 results/followup/summary.csv 中是否已存在
#   对应 (method, placement, dataset, seed) 行自动跳过(无需任何 flag 文件)。
# 后台运行: nohup bash scripts/run_followup.sh > followup.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DATA=./data; OUT=./results/followup
NW=${NW:-8}     # AutoDL 实例有 24 vCPU,8 个 worker 明显快于默认 2
mkdir -p "$DATA" "$OUT"

already_done () {   # $1=method(base)  $2=placement  $3=dataset  $4=seed  -> 命中(已在 summary.csv)返回 0
  local csv="$OUT/summary.csv"
  [ -f "$csv" ] || return 1
  awk -F, -v m="$1" -v p="$2" -v d="$3" -v s="$4" \
    'NR>1 && $1==m && $2==p && $3==d && $4==s {found=1} END{exit !found}' "$csv"
}

run () {   # $1=method  $2=dataset  $3=epochs  $4=seed  [$5=blocks  $6=label]
  local m="$1" ds="$2" ep="$3" sd="$4" blocks="${5:-}" label="${6:-}"
  local base pos
  base="${m%%-*}"
  if [ -n "$blocks" ]; then pos="$label";
  elif [ "$m" = "$base" ]; then pos="all"; else pos="${m#*-}"; fi
  if already_done "$base" "$pos" "$ds" "$sd"; then echo "== skip $base/$pos $ds seed$sd(summary.csv 已有)"; return 0; fi
  echo "===== $base/$pos / $ds / seed $sd ($ep ep) ====="
  if [ -n "$blocks" ]; then
    python train.py --method "$m" --dataset "$ds" --epochs "$ep" --seed "$sd" \
      --blocks "$blocks" --placement-label "$label" \
      --num-workers "$NW" --data-root "$DATA" --out-dir "$OUT"
  else
    python train.py --method "$m" --dataset "$ds" --epochs "$ep" --seed "$sd" \
      --num-workers "$NW" --data-root "$DATA" --out-dir "$OUT"
  fi
}

echo "############  A 部分: DTD(47 类纹理,30 轮,与 Flowers/Pets 同协议)  ############"
for sd in 42 43 44; do
  for m in linear bitfit full lora ssf; do
    run "$m" dtd 30 "$sd"
  done
  for b in lora ssf; do
    for p in early mid late even; do
      run "$b-$p" dtd 30 "$sd"
    done
  done
done

echo "############  B 部分: Fisher 自动选层(cifar100/flowers/dtd × 3 种子)  ############"
declare -A EP=( [cifar100]=20 [flowers]=30 [dtd]=30 )
for ds in cifar100 flowers dtd; do
  for sd in 42 43 44; do
    json="$OUT/fisher_${ds}_seed${sd}.json"
    if [ ! -f "$json" ]; then
      echo "----- fisher_select: $ds seed$sd -----"
      python scripts/fisher_select.py --dataset "$ds" --seed "$sd" --k 4 \
        --data-root "$DATA" --out-dir "$OUT" --num-workers "$NW" || exit 1
    fi
    BLK=$(python -c "import json;print(','.join(map(str,json.load(open('$json'))['selected'])))")
    echo "----- $ds seed$sd: selected blocks = $BLK -----"
    run lora "$ds" "${EP[$ds]}" "$sd" "$BLK" auto
  done
done

echo "ALL_FOLLOWUP_DONE  ->  接着跑: python scripts/summarize_followup.py"
