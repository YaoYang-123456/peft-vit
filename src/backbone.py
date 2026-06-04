"""Backbone: ViT-B/16 pre-trained on ImageNet-21k (the standard PEFT-for-vision model)."""
import timm

DEFAULT_MODEL = "vit_base_patch16_224.augreg_in21k"


def create_model(num_classes, drop_path_rate=0.0, model_name=DEFAULT_MODEL):
    """Create an IN-21k ViT-B/16 with a fresh `num_classes` head.

    drop_path (stochastic depth) is only useful when the backbone is trainable
    (full fine-tuning). For frozen-backbone methods keep it at 0.0.
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
    )
    return model


def get_default_data_config(model_name=DEFAULT_MODEL):
    """Resolve the model's expected preprocessing (mean/std/size) WITHOUT
    downloading the pretrained weights."""
    m = timm.create_model(model_name, pretrained=False, num_classes=0)
    return timm.data.resolve_model_data_config(m)
