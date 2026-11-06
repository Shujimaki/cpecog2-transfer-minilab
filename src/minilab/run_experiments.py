"""Run the mini-lab sweep: 4 strategies x 4 training-set sizes.

    uv run python -m minilab.run_experiments            # full sweep (16 runs)
    uv run python -m minilab.run_experiments --quick    # smoke test, a few epochs per model
    uv run python -m minilab.run_experiments --seeds 0 1 2
    uv run python -m minilab.run_experiments --strategies scratch full --fractions 1.0
"""

import argparse

import pandas as pd
import torch

from minilab import config as C
from minilab import plots
from minilab.data import build_cache, make_loaders
from minilab.metrics import evaluate
from minilab.train import fit


def largest_fraction_table(results: pd.DataFrame) -> pd.DataFrame:
    top = results[results.fraction == results.fraction.max()]
    table = top.groupby("strategy", sort=False).agg(
        accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
        trainable_params=("trainable_params", "first"), train_time_s=("train_time_s", "mean"))
    table.index = [plots.LABELS[s] for s in table.index]
    # to_markdown prints int columns of a mixed frame as floats (10245.000)
    table["trainable_params"] = table["trainable_params"].map("{:,}".format)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", default=C.STRATEGIES, choices=C.STRATEGIES)
    parser.add_argument("--fractions", nargs="+", type=float, default=C.TRAIN_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[C.SEED])
    parser.add_argument("--quick", action="store_true", help="3 epochs per model, 25%% of the data only")
    args = parser.parse_args()

    out_dir = C.RESULTS_DIR / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = [0.25] if args.quick else args.fractions
    max_epochs = C.WARMUP_EPOCHS + 1 if args.quick else None

    device = C.get_device()
    cache = build_cache()
    print(f"Device: {device}. Split sizes: " +
          ", ".join(f"{k} {len(v[1])}" for k, v in cache.items()))

    rows, best = [], None
    for seed in args.seeds:
        for fraction in fractions:
            for strategy in args.strategies:
                torch.manual_seed(seed)
                train_loader, val_loader, test_loader = make_loaders(cache, fraction, seed)
                n_train = len(train_loader.dataset)
                print(f"\n[{strategy}] fraction {fraction:.0%} ({n_train} images), seed {seed}")
                model, trainable, seconds, history = fit(strategy, train_loader, val_loader, device, max_epochs)
                scores = evaluate(model, test_loader, device)
                row = {"strategy": strategy, "fraction": fraction, "n_train": n_train, "seed": seed,
                       "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
                       "trainable_params": trainable, "train_time_s": seconds, "epochs": len(history["val_loss"])}
                rows.append(row)
                print(f"  test accuracy {row['accuracy']:.3f}  macro-F1 {row['macro_f1']:.3f}  "
                      f"trainable {trainable:,}  time {seconds:.1f}s")
                pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

                if best is None or row["accuracy"] > best[0]["accuracy"]:
                    best = (row, scores["confusion"], history)

    results = pd.DataFrame(rows)
    plots.accuracy_vs_size(results, out_dir / "accuracy_vs_size.png")

    table = largest_fraction_table(results)
    (out_dir / "table.md").write_text(table.to_markdown(floatfmt=".3f") + "\n")
    print("\n" + table.to_string())

    row, cm, history = best
    name = f"{plots.LABELS[row['strategy']]}, {row['fraction']:.0%} of train"
    plots.confusion(cm, f"Confusion matrix: {name}", out_dir / "confusion_best.png")
    plots.training_curves(history, f"Training curves: {name}", out_dir / "curves_best.png")
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
