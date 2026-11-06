"""The four models of the mini-lab.

scratch             a small CNN with random weights, every weight trains
feature_extraction  ResNet-50, backbone frozen, only the new fc head trains
partial             ResNet-50, fc + layer4 train (after a head warm-up)
full                ResNet-50, every layer trains (after a head warm-up)
"""

import torch.nn as nn
from torchvision import models

from minilab import config as C


def small_cnn(num_classes: int = len(C.CLASSES)) -> nn.Module:
    """Three Conv-BN-ReLU-MaxPool blocks, then global average pooling and one Linear layer."""

    def block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )


def resnet50_with_new_head(num_classes: int = len(C.CLASSES)) -> nn.Module:
    """ImageNet ResNet-50 with every weight frozen and a fresh 2048 x C fc layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new layers are trainable by default
    return model


def unfreeze(model: nn.Module, strategy: str) -> list[dict]:
    """Unfreeze the layers a strategy fine-tunes. Returns optimizer param groups.

    Called after the head warm-up. The backbone gets a learning rate 10x smaller than the head.
    """
    if strategy == "partial":
        backbone = list(model.layer4.parameters())
    elif strategy == "full":
        backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    else:
        raise ValueError(f"nothing to unfreeze for {strategy!r}")
    for p in backbone:
        p.requires_grad = True
    return [
        {"params": backbone, "lr": C.BACKBONE_LR},
        {"params": model.fc.parameters(), "lr": C.HEAD_LR},
    ]


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep BatchNorm running statistics fixed (the lecture's BatchNorm caveat).

    Call this after every model.train(). With small batches of new data, updating the
    pretrained running mean and variance makes them noisy and hurts accuracy.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
