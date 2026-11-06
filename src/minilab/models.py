import torch.nn as nn
from torchvision import models

from minilab import config as C


def small_cnn(num_classes: int = len(C.CLASSES)) -> nn.Module:
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


def frozen_resnet50_with_new_head(num_classes: int = len(C.CLASSES)) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_param_groups(model: nn.Module, strategy: str) -> list[dict]:
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


def set_train_mode(model: nn.Module, freeze_bn_stats: bool) -> None:
    model.train()
    if freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
