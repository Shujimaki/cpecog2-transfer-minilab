"""One training run: build the model for a strategy, fit it, and time it."""

import copy
import time

import torch
import torch.nn as nn

from minilab import config as C
from minilab.data import prepare_batch
from minilab.models import count_trainable, freeze_batchnorm, resnet50_with_new_head, small_cnn, unfreeze


def _run_epoch(model, loader, device, optimizer=None, keep_bn_frozen=False):
    """Train for one epoch if an optimizer is given, otherwise just measure the loss."""
    training = optimizer is not None
    model.train(training)
    if training and keep_bn_frozen:
        freeze_batchnorm(model)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = prepare_batch(x, device, augment=training), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def fit(strategy: str, train_loader, val_loader, device, max_epochs: int | None = None):
    """Train one model with early stopping on validation loss.

    Returns (model with the best weights, trainable parameter count, seconds, history).
    """
    max_epochs = max_epochs or C.MAX_EPOCHS[strategy]
    is_transfer = strategy != "scratch"

    if is_transfer:
        model = resnet50_with_new_head().to(device)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=C.HEAD_LR)
    else:
        model = small_cnn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=C.SCRATCH_LR)

    # Partial and full fine-tuning: warm up the random head first, then unfreeze.
    # A random head sends large gradients that can wreck pretrained weights.
    unfreeze_at = C.WARMUP_EPOCHS if strategy in ("partial", "full") else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    trainable = count_trainable(model)

    _sync(device)
    start = time.perf_counter()
    for epoch in range(max_epochs):
        if epoch == unfreeze_at:
            optimizer = torch.optim.Adam(unfreeze(model, strategy))
            trainable = count_trainable(model)
            bad_epochs = 0  # give the fine-tuning stage its own patience

        tr_loss, tr_acc = _run_epoch(model, train_loader, device, optimizer, keep_bn_frozen=is_transfer)
        va_loss, va_acc = _run_epoch(model, val_loader, device)
        for key, value in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(value)
        print(f"    epoch {epoch + 1:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_loss < best_loss:
            best_loss, best_state, bad_epochs = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            # Never stop during the warm-up: the fine-tuning stage has not started yet.
            warming_up = unfreeze_at is not None and epoch < unfreeze_at
            if bad_epochs >= C.PATIENCE and not warming_up:
                break
    _sync(device)
    seconds = time.perf_counter() - start

    model.load_state_dict(best_state)
    return model, trainable, seconds, history
