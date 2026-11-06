# Transfer Learning Mini-Lab

This lab trains four models on a 5-class flower dataset:

- a small CNN from scratch
- ResNet-50 feature extraction
- ResNet-50 partial fine-tuning
- ResNet-50 full fine-tuning

Each model is trained on 10%, 25%, 50%, and 100% of the training data. The write-up is in [REPORT.md](REPORT.md).

## Dataset

The dataset is TF Flowers, with 300 images per class and a 900/300/300 train/val/test split. It downloads automatically on the first run (about 220 MB).

## Run

```bash
uv sync
```

```bash
uv run python -m minilab.run_experiments
```

Add `--quick` for a short test run. Results go to `results/`:

| File | Contents |
|---|---|
| `results.csv` | All runs |
| `table.md` | Results on 100% of the data |
| `accuracy_vs_size.png` | Accuracy vs training-set size |
| `confusion_best.png`, `curves_best.png` | Best model's confusion matrix and training curves |

## Code

| File | What it does |
|---|---|
| `config.py` | All settings |
| `data.py` | Downloads, splits, and loads the data |
| `models.py` | Builds the four models |
| `train.py` | Trains one model |
| `metrics.py` | Accuracy, macro-F1, confusion matrix |
| `plots.py` | Figures |
| `run_experiments.py` | Runs every model at every size |

## Settings

| Setting | Value |
|---|---|
| Input | 224x224, ImageNet normalization |
| New head | 2048 to 5 (10,245 params) |
| Head learning rate | 1e-3 |
| Backbone learning rate | 1e-4 |
| Head warm-up before unfreezing | 2 epochs |
| BatchNorm | Frozen in all ResNet-50 models |
| Early stopping | Patience 3 on validation loss |

## Keras to PyTorch

| Keras (LBYCPH3) | PyTorch (here) |
|---|---|
| `base.trainable = False` | `p.requires_grad = False` |
| `base(x, training=False)` | `set_train_mode(model, freeze_bn_stats=True)` |
| Recompile with a lower learning rate | New optimizer with per-layer learning rates |

PyTorch is used because it runs on the Mac GPU (MPS). TensorFlow would run on the CPU only.
