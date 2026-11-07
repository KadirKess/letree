"""Train a correction model, its clean-only control, or one stage of staged training.

uv run python scripts/train.py DATASET RUN_DIRECTORY --max-epochs N
uv run python scripts/train.py DATASET OBSERVED_RUN --max-epochs N --stage observed
uv run python scripts/train.py DATASET RUN_DIRECTORY --max-epochs N \
    --stage frozen --observed-run OBSERVED_RUN
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
    parser.add_argument("--precision", choices=("fp32", "tf32"), default="fp32")
    parser.add_argument("--compile", choices=("none", "model", "step"), default="none")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument("--aggregations", nargs="*", default=["max", "min"])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--observed-steps", type=int, default=10)
    parser.add_argument("--correction-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage", choices=("joint", "observed", "frozen"), default="joint"
    )
    parser.add_argument("--observed-run")
    parser.add_argument("--zone-pointer", action="store_true")
    parser.add_argument("--separating-weight", type=float, default=1.0)
    parser.add_argument("--backbone", action="store_true")
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
        precision=arguments.precision,
        compile=arguments.compile,
        compile_mode=arguments.compile_mode,
        observed_steps=arguments.observed_steps,
        correction_steps=arguments.correction_steps,
        plateau_patience=arguments.plateau_patience,
        early_stopping_patience=arguments.early_stopping_patience,
        batch_size=arguments.batch_size,
        num_workers=arguments.workers,
        seed=arguments.seed,
        stage=arguments.stage,
        observed_run=arguments.observed_run,
        zone_pointer=arguments.zone_pointer,
        separating_weight=arguments.separating_weight,
        backbone=arguments.backbone,
    )
    best = fit(settings, arguments.run_directory, resume=arguments.resume)
    print(f"best epoch {best['epoch']}")
    for name, value in best["validation"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
