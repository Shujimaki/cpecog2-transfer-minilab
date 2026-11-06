import random
import tarfile
import urllib.request

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from minilab import config as C


def download() -> None:
    raw = C.DATA_DIR / "raw"
    if (raw / "flower_photos").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "flower_photos.tgz"
    print(f"Downloading {C.DATASET_URL} ...")
    urllib.request.urlretrieve(C.DATASET_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(raw, filter="data")
    archive.unlink()


def _load_image(path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((C.IMAGE_SIZE, C.IMAGE_SIZE))
    return torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(C.IMAGE_SIZE, C.IMAGE_SIZE, 3)


def build_cache() -> dict:
    cache_file = C.DATA_DIR / "cache.pt"
    if cache_file.exists():
        return torch.load(cache_file)

    download()
    rng = random.Random(C.SEED)
    paths, labels = [], []
    for label, name in enumerate(C.CLASSES):
        files = sorted((C.DATA_DIR / "raw" / "flower_photos" / name).glob("*.jpg"))
        print(f"  {name}: {len(files)} images available, keeping {C.IMAGES_PER_CLASS}")
        paths += rng.sample(files, C.IMAGES_PER_CLASS)
        labels += [label] * C.IMAGES_PER_CLASS

    train_frac, val_frac, test_frac = C.SPLIT
    p_train, p_rest, y_train, y_rest = train_test_split(
        paths, labels, train_size=train_frac, stratify=labels, random_state=C.SEED)
    p_val, p_test, y_val, y_test = train_test_split(
        p_rest, y_rest, test_size=test_frac / (val_frac + test_frac), stratify=y_rest, random_state=C.SEED)

    print("Decoding and resizing images (first run only) ...")
    cache = {}
    for split, (p, y) in {"train": (p_train, y_train), "val": (p_val, y_val), "test": (p_test, y_test)}.items():
        x = torch.stack([_load_image(f) for f in p]).permute(0, 3, 1, 2).contiguous()
        cache[split] = (x, torch.tensor(y))
    torch.save(cache, cache_file)
    return cache


def nested_train_subset(x: torch.Tensor, y: torch.Tensor, fraction: float, seed: int = C.SEED):
    if fraction >= 1.0:
        return x, y
    g = torch.Generator().manual_seed(seed)
    keep = []
    for label in y.unique():
        idx = (y == label).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        keep.append(idx[: max(1, round(fraction * len(idx)))])
    keep = torch.cat(keep)
    return x[keep], y[keep]


_MEAN = torch.tensor(C.IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(C.IMAGENET_STD).view(1, 3, 1, 1)


def prepare_batch(x_uint8: torch.Tensor, device, augment: bool) -> torch.Tensor:
    x = x_uint8.to(device).float() / 255.0
    if augment:
        flip = torch.rand(x.shape[0], device=device) < 0.5
        x[flip] = x[flip].flip(-1)
    return (x - _MEAN.to(device)) / _STD.to(device)


def make_loaders(cache: dict, fraction: float, seed: int = C.SEED):
    x_train, y_train = nested_train_subset(*cache["train"], fraction, seed)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(x_train, y_train), batch_size=C.BATCH_SIZE, shuffle=True, generator=g)
    val = DataLoader(TensorDataset(*cache["val"]), batch_size=C.BATCH_SIZE)
    test = DataLoader(TensorDataset(*cache["test"]), batch_size=C.BATCH_SIZE)
    return train, val, test
