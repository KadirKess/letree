"""Decode and score a trained checkpoint on one split.

uv run python scripts/evaluate.py RUN_DIRECTORY DATASET --split test
"""

import argparse

from letree.evaluation.checkpoint import run_evaluation
from letree.evaluation.records import SOURCES


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    # Without CUDA, Flex and compiled kernels are replaced by their plain versions
    parser.add_argument("--attention", choices=("dense", "flex"))
    parser.add_argument("--compile", choices=("none", "model", "step"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    summary = run_evaluation(
        arguments.run_directory,
        arguments.dataset,
        arguments.split,
        arguments.checkpoint,
        arguments.batch_size,
        arguments.workers,
        **{
            name: value
            for name, value in (
                ("attention", arguments.attention),
                ("compile", arguments.compile),
            )
            if value is not None
        },
    )
    names = (
        "coverage",
        "exact_recovery",
        "root_claims/sample_mean",
        "all/misassigned_pixels/sample_mean",
    )
    for group in ("clean", "noisy"):
        for source in SOURCES:
            prefix = f"{source}/{group}/every"
            values = [summary.get(f"{prefix}/{name}", float("nan")) for name in names]
            print(group, source, *(f"{name} {v:.4f}" for name, v in zip(names, values)))


if __name__ == "__main__":
    main()
