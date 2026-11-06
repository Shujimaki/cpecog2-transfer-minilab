import copy
import time
from typing import NamedTuple

import torch
import torch.nn as nn

from minilab import config as C
from minilab.data import prepare_batch
from minilab.models import (count_trainable, frozen_resnet50_with_new_head, set_train_mode, small_cnn,
                            unfreeze_param_groups)

loss_fn = nn.CrossEntropyLoss()


class FitResult(NamedTuple):
    model: nn.Module
    trainable_params: int
    seconds: float
    history: dict


def train_epoch(model, loader, device, optimizer, freeze_bn_stats: bool):
    set_train_mode(model, freeze_bn_stats)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = prepare_batch(x, device, augment=True), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = prepare_batch(x, device, augment=False), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def fit(strategy: str, train_loader, val_loader, device, max_epochs: int | None = None) -> FitResult:
    max_epochs = max_epochs or C.MAX_EPOCHS[strategy]
    is_transfer = strategy != "scratch"

    if is_transfer:
        model = frozen_resnet50_with_new_head().to(device)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=C.HEAD_LR)
    else:
        model = small_cnn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=C.SCRATCH_LR)

    unfreeze_at = C.WARMUP_EPOCHS if strategy in ("partial", "full") else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    trainable = count_trainable(model)

    _sync(device)
    start = time.perf_counter()
    for epoch in range(max_epochs):
        if epoch == unfreeze_at:
            optimizer = torch.optim.Adam(unfreeze_param_groups(model, strategy))
            trainable = count_trainable(model)
            bad_epochs = 0

        tr_loss, tr_acc = train_epoch(model, train_loader, device, optimizer, freeze_bn_stats=is_transfer)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        for key, value in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(value)
        print(f"    epoch {epoch + 1:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_loss < best_loss:
            best_loss, best_state, bad_epochs = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            warming_up = unfreeze_at is not None and epoch < unfreeze_at
            if bad_epochs >= C.PATIENCE and not warming_up:
                break
    _sync(device)
    seconds = time.perf_counter() - start

    model.load_state_dict(best_state)
    return FitResult(model, trainable, seconds, history)
