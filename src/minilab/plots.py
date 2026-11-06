import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from minilab import config as C

LABELS = {
    "scratch": "Small CNN (scratch)",
    "feature_extraction": "ResNet-50 feature extraction",
    "partial": "ResNet-50 partial fine-tuning",
    "full": "ResNet-50 full fine-tuning",
}


def accuracy_vs_size(results: pd.DataFrame, path) -> None:
    summary = results.groupby(["strategy", "fraction"]).agg(
        acc=("accuracy", "mean"), std=("accuracy", "std"), n_train=("n_train", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy in C.STRATEGIES:
        s = summary[summary.strategy == strategy].sort_values("fraction")
        if s.empty:
            continue
        ax.errorbar(s.fraction * 100, s.acc * 100, yerr=s["std"].fillna(0) * 100,
                    marker="o", capsize=3, label=LABELS[strategy])
    ax.set_xlabel("Training-set size (% of the 900 training images)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs training-set size (TF Flowers, 5 classes)")
    ax.set_xticks([f * 100 for f in C.TRAIN_FRACTIONS])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(history: dict, title: str, path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    panels = [("train_loss", "Training loss"), ("train_acc", "Training accuracy"),
              ("val_loss", "Validation loss"), ("val_acc", "Validation accuracy")]
    for ax, (key, name) in zip(axes.flat, panels):
        ax.plot(epochs, history[key], marker="o")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion(cm, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASSES).plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
