"""Configure which parameters are trainable for each fine-tuning method.

Phase 1 implements the two baselines (linear probing and full fine-tuning).
Phase 2 will add BitFit, LoRA, AdaptFormer and SSF here -- the rest of the
training harness (data, engine, logging) stays unchanged.
"""


def configure_method(model, method):
    method = method.lower()

    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)

    elif method == "linear":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    else:
        raise NotImplementedError(
            f"Method '{method}' is not implemented yet (coming in Phase 2). "
            f"Available now: 'linear', 'full'."
        )

    return model
