import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from config import Config as C


def build_deeplab_model() -> nn.Module:
    """
    DeepLabV3-ResNet101 with classifier head sized for C.num_classes.
    Uses ImageNet-pretrained weights for the backbone by default.
    """
    model = deeplabv3_resnet101(weights="DEFAULT")
    # ResNet-101 backbone → 2048 channels into classifier head
    model.classifier = DeepLabHead(2048, C.num_classes)
    return model
