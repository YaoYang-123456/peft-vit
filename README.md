# PEFT-ViT: Parameter-Efficient Fine-Tuning of Vision Transformers

A clean, from-scratch study comparing parameter-efficient fine-tuning (PEFT) methods
on a ViT-B/16 backbone pre-trained on ImageNet-21k, across several downstream image
classification datasets.

Backbone: `vit_base_patch16_224.augreg_in21k` (timm). The classification head is
replaced per dataset; everything else is frozen for the PEFT methods.

## Status

- **Phase 1 (current):** training harness + two baselines — **linear probing** and **full fine-tuning**.
- **Phase 2 (next):** BitFit, LoRA, AdaptFormer, SSF, plus a depth-aware LoRA+SSF hybrid.

## Environment

```bash
pip install -r requirements.txt   # on Colab: only `pip install -U timm` is needed
```

Python 3.10+, PyTorch 2.x, a CUDA GPU (developed on Google Colab Pro, T4/L4/A100).

## Datasets

Downloaded automatically by torchvision on first run (`--data-root`).

| Dataset | Classes | Train / Val / Test |
|---|---|---|
| CIFAR-100 | 100 | 45 000 / 5 000 / 10 000 (val carved from train) |
| Oxford Flowers-102 | 102 | 1 020 / 1 020 / 6 149 (official splits) |
| Oxford-IIIT Pets | 37 | ~3 312 / ~368 / 3 669 (val carved from trainval) |
| DTD | 47 | 1 880 / 1 880 / 1 880 (partition 1) |

Hyperparameters are selected on the validation set; the test set is evaluated exactly
once, using the best-validation checkpoint.

## Unified protocol

Same preprocessing and schedule for every method, so differences reflect the method
and not the recipe:

- train aug: `RandomResizedCrop(224, scale=(0.5,1.0))` + horizontal flip
- eval: `Resize(256)` + `CenterCrop(224)`
- optimizer AdamW, cosine LR with 10-epoch linear warmup, mixed precision (fp16)
- input 224×224, normalization `(0.5, 0.5, 0.5)`

## Run

```bash
# baselines
python train.py --method linear --dataset flowers  --epochs 100
python train.py --method full   --dataset flowers  --epochs 100
python train.py --method linear --dataset pets     --epochs 100
python train.py --method full   --dataset pets     --epochs 100
python train.py --method linear --dataset cifar100 --epochs 50
python train.py --method full   --dataset cifar100 --epochs 50
```

Per-run epoch curves are written to `results/<method>_<dataset>_seed<seed>.csv`,
and a one-line summary (trainable params, best-val, test accuracy) is appended to
`results/summary.csv`.

## Results

(to be filled in — main comparison table + accuracy-vs-trainable-params Pareto plot)
