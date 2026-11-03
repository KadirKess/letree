"""Train a correction model, or its clean-only control.

uv run python scripts/train.py DATASET RUN_DIRECTORY --max-epochs N
"""

import argparse

from letree.experiments.train import RunSettings, fit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("run_directory")
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attention", choices=("dense", "flex"), default="dense")
    parser.add_argument("--aggregations", nargs="*", default=["max", "min"])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--observed-steps", type=int, default=10)
    parser.add_argument("--correction-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    settings = RunSettings(
        dataset_root=arguments.dataset,
        max_epochs=arguments.max_epochs,
        include_noisy=not arguments.clean_only,
        width=arguments.width,
        aggregations=tuple(arguments.aggregations),
        attention=arguments.attention,
        observed_steps=arguments.observed_steps,
        correction_steps=arguments.correction_steps,
        plateau_patience=arguments.plateau_patience,
        early_stopping_patience=arguments.early_stopping_patience,
        batch_size=arguments.batch_size,
        num_workers=arguments.workers,
        seed=arguments.seed,
    )
    best = fit(settings, arguments.run_directory, resume=arguments.resume)
    print(f"best epoch {best['epoch']}")
    for name, value in best["validation"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
