#!/bin/bash
# 追加实验(报告 §3.8/§3.9 预定内容),全部写入 results/followup/,不动主结果。
#   A 部分: 第四个数据集 DTD(纹理,语义偏移大但低级统计接近自然图像)
#           - 五种方法整体对比 × 3 种子            = 15 runs
#           - LoRA/SSF × {early,mid,late,even} × 3 种子 = 24 runs
#   B 部分: Fisher 自动选层("深度感知 PEFT"的自动化)
#           - 每个 (数据集, 种子) 先跑 fisher_select.py 选出 4 个 Block,
#             再用所选 Block 训练 LoRA(r=8,placement 标签 = auto)
#           - cifar100 / flowers / dtd × 3 种子      = 9 selections + 9 runs
# 预计总耗时:RTX 5090D 上约 3.5–4.5 小时。
# 已完成的 run 自动跳过(results/followup/done_*.flag)。
# 后台运行: nohup bash scripts/run_followup.sh > followup.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DATA=./data; OUT=./results/followup
NW=${NW:-8}     # AutoDL 实例有 24 vCPU,8 个 worker 明显快于默认 2
mkdir -p "$DATA" "$OUT"

run () {   # $1=method  $2=dataset  $3=epochs  $4=seed  [$5=blocks  $6=label]
  local m="$1" ds="$2" ep="$3" sd="$4" blocks="${5:-}" label="${6:-}"
  local base pos
  base="${m%%-*}"
  if [ -n "$blocks" ]; then pos="$label";
  elif [ "$m" = "$base" ]; then pos="all"; else pos="${m#*-}"; fi
  local flag="$OUT/done_${base}_${pos}_${ds}_${sd}.flag"
  if [ -f "$flag" ]; then echo "== skip $base/$pos $ds seed$sd"; return 0; fi
  echo "===== $base/$pos / $ds / seed $sd ($ep ep) ====="
  if [ -n "$blocks" ]; then
    python train.py --method "$m" --dataset "$ds" --epochs "$ep" --seed "$sd" \
      --blocks "$blocks" --placement-label "$label" \
      --num-workers "$NW" --data-root "$DATA" --out-dir "$OUT" && touch "$flag"
  else
    python train.py --method "$m" --dataset "$ds" --epochs "$ep" --seed "$sd" \
      --num-workers "$NW" --data-root "$DATA" --out-dir "$OUT" && touch "$flag"
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
