# deeplab_froth/models/deeplabv3plus.py
import torch.nn as nn

from config import Config as C

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation_models_pytorch is required for DeepLabV3+.\n"
        "Install with:\n\n"
        "  pip install -U segmentation-models-pytorch timm\n"
    ) from e


def build_deeplabv3plus_model() -> nn.Module:
    """
    DeepLabV3+ with ResNet-152 backbone, output stride 8.
    Uses ImageNet-pretrained encoder weights.
    Head is sized for C.num_classes.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet152",
        encoder_weights="imagenet",
        encoder_output_stride=8,   # OS=8; change to 16 if OOM
        in_channels=3,
        classes=C.num_classes,
        activation=None,
    )
    return model
