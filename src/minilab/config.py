from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMAGES_PER_CLASS = 300
SPLIT = (0.6, 0.2, 0.2)
IMAGE_SIZE = 224

# Pretrained torchvision weights expect inputs normalized this way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STRATEGIES = ["scratch", "feature_extraction", "partial", "full"]
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

BATCH_SIZE = 32
MAX_EPOCHS = {"scratch": 30, "feature_extraction": 40, "partial": 10, "full": 10}
WARMUP_EPOCHS = 2
PATIENCE = 3
HEAD_LR = 1e-3
BACKBONE_LR = HEAD_LR / 10
SCRATCH_LR = 1e-3
SEED = 0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
