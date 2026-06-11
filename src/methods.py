"""Configure which parameters are trainable for each fine-tuning method.

Phase 1: linear probing, full fine-tuning.
Phase 2: BitFit, LoRA, SSF -- clean from-scratch reimplementations of the
published methods (cited below), all plugged into the SAME training harness.
The harness (data / engine / logging) does not change; each method only decides
which parameters are trainable and, for LoRA/SSF, injects small trainable modules.

References (reimplemented from the papers, not copied from their code):
  - BitFit: Ben-Zaken, Ravfogel, Goldberg. "BitFit: Simple Parameter-efficient
            Fine-tuning for Transformer-based Masked Language-models." ACL 2022.
  - LoRA:   Hu, Shen, Wallis, et al. "LoRA: Low-Rank Adaptation of Large
            Language Models." ICLR 2022.
  - SSF:    Lian, Zhou, Feng, Wang. "Scaling & Shifting Your Features: A New
            Baseline for Efficient Model Tuning." NeurIPS 2022.
"""
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Small trainable modules used by LoRA and SSF
# --------------------------------------------------------------------------- #
class LoRAQKV(nn.Module):
    """Wrap the (frozen) fused qkv Linear of a timm attention block and add a
    trainable low-rank update to its output:  out = W0 @ x + (B @ A) @ x * (a/r).

    timm fuses q, k, v into one Linear(dim -> 3*dim).  We add a single low-rank
    branch over the whole fused projection (same parameter budget as adapting q
    and v separately).  A is kaiming-initialised and B is zero, so the branch
    starts as an exact identity and the model begins at its pre-trained behaviour.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + delta * self.scaling


class SSF(nn.Module):
    """Per-channel learnable scale and shift:  y = scale * x + shift.

    Initialised to identity (scale=1, shift=0) so it does not disturb the
    pre-trained features at the start of training.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.scale + self.shift


# --------------------------------------------------------------------------- #
#  Injection helpers (light module surgery on a timm VisionTransformer)
# --------------------------------------------------------------------------- #
def _placement_to_block_ids(placement, n_blocks):
    """Map a placement keyword to the indices of the blocks to adapt.
    early/mid/late take one contiguous third of the network; 'all' takes every
    block.  This lets us ask "where should a fixed budget go?" at matched cost.
    """
    placement = (placement or "all").lower()
    if placement == "all":
        return list(range(n_blocks))
    t = n_blocks // 3                      # 4 for ViT-B (12 blocks)
    if placement == "even":                # spread: every 3rd block -> 0,3,6,9
        return list(range(0, n_blocks, max(1, n_blocks // 4)))[:4]
    spans = {"early":    range(0, t),
             "mid":      range(t, 2 * t),
             "late":     range(2 * t, n_blocks),
             "earlymid": range(0, 2 * t)}
    if placement not in spans:
        raise ValueError(f"unknown placement '{placement}' (use all/early/mid/late/earlymid/even)")
    return list(spans[placement])


def _inject_lora(model, r=8, alpha=16, block_ids=None):
    """Replace the fused qkv Linear with a LoRA-wrapped version in the chosen blocks."""
    if block_ids is None:
        block_ids = range(len(model.blocks))
    block_ids = set(block_ids)
    for i, blk in enumerate(model.blocks):
        if i in block_ids:
            blk.attn.qkv = LoRAQKV(blk.attn.qkv, r=r, alpha=alpha)


def _wrap_ssf(parent, attr, dim):
    """parent.attr = Sequential(parent.attr, SSF(dim))  -- keeps the original op
    and applies a learnable scale/shift to its output."""
    orig = getattr(parent, attr)
    setattr(parent, attr, nn.Sequential(orig, SSF(dim)))


def _inject_ssf(model, block_ids=None):
    """Insert an SSF transform after the main linear/norm ops inside the chosen blocks."""
    d = model.embed_dim                                  # 768 for ViT-B/16
    qkv_out = model.blocks[0].attn.qkv.out_features      # 3 * 768 = 2304
    mlp_hidden = model.blocks[0].mlp.fc1.out_features    # 3072
    if block_ids is None:
        block_ids = range(len(model.blocks))
    block_ids = set(block_ids)
    for i, blk in enumerate(model.blocks):
        if i not in block_ids:
            continue
        _wrap_ssf(blk, "norm1", d)
        _wrap_ssf(blk.attn, "qkv", qkv_out)
        _wrap_ssf(blk.attn, "proj", d)
        _wrap_ssf(blk, "norm2", d)
        _wrap_ssf(blk.mlp, "fc1", mlp_hidden)
        _wrap_ssf(blk.mlp, "fc2", d)


# --------------------------------------------------------------------------- #
#  Public entry point (called once by train.py after the model is built)
# --------------------------------------------------------------------------- #
def configure_method(model, method, block_ids=None):
    """block_ids: optional explicit list of block indices for LoRA/SSF.
    When given (e.g. by an automatic layer-selection procedure), it overrides
    any placement suffix in `method`."""
    method = method.lower()
    # train.py moves the model to the GPU BEFORE calling this; any modules we
    # inject below (LoRA/SSF) are created on CPU, so capture the device now and
    # move the whole model back onto it at the end.
    device = next(model.parameters()).device

    # method may carry placement/rank suffixes, e.g. "lora-late", "lora-r12-early".
    toks = method.split("-")
    base = toks[0]
    placement, rank = "all", 8
    for t in toks[1:]:
        if len(t) > 1 and t[0] == "r" and t[1:].isdigit():
            rank = int(t[1:])
        else:
            placement = t

    if base == "full":
        for p in model.parameters():
            p.requires_grad_(True)

    elif base == "linear":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    elif base == "bitfit":
        # train every bias term + the classification head
        for p in model.parameters():
            p.requires_grad_(False)
        for n, p in model.named_parameters():
            if n.endswith(".bias"):
                p.requires_grad_(True)
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    elif base == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        ids = (sorted(set(int(b) for b in block_ids)) if block_ids is not None
               else _placement_to_block_ids(placement, len(model.blocks)))
        _inject_lora(model, r=rank, alpha=2 * rank, block_ids=ids)   # trainable low-rank params
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    elif base == "ssf":
        for p in model.parameters():
            p.requires_grad_(False)
        ids = (sorted(set(int(b) for b in block_ids)) if block_ids is not None
               else _placement_to_block_ids(placement, len(model.blocks)))
        _inject_ssf(model, block_ids=ids)        # trainable scale/shift params
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    else:
        raise NotImplementedError(
            f"Method '{method}' is not implemented. "
            f"Available: linear, full, bitfit, lora, ssf."
        )

    model.to(device)   # ensure newly-injected LoRA/SSF params are on the model's device
    return model
