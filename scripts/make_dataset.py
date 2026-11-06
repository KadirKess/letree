"""Build correction families from a clean image source.

uv run python scripts/make_dataset.py SOURCE DATASET --noise salt_and_pepper
"""

import argparse
import os

from letree.component_tree.types import ComponentTreeKind
from letree.data.families import generate_dataset
from letree.generation.noise import NOISE_KIND_NAMES, AdditiveGaussianNoise

NOISE_TYPES = {name: noise_type for noise_type, name in NOISE_KIND_NAMES.items()}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("dataset")
    parser.add_argument(
        "--noise", nargs="+", choices=NOISE_TYPES, default=list(NOISE_TYPES)
    )
    parser.add_argument("--noisy", type=int, default=2)
    parser.add_argument(
        "--gaussian-std",
        type=float,
        default=AdditiveGaussianNoise.standard_deviation,
    )
    parser.add_argument(
        "--tree", choices=[kind.value for kind in ComponentTreeKind], default="max_tree"
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def make_noise(name: str, arguments: argparse.Namespace):
    if name == "additive_gaussian":
        return AdditiveGaussianNoise(standard_deviation=arguments.gaussian_std)
    return NOISE_TYPES[name]()


def main() -> None:
    arguments = parse_arguments()
    # Training images get a few kinds each, evaluation images every kind once,
    # so each kind is scored on the same validation and test images
    kind_count = len(arguments.noise)
    noisy_counts = {"train": arguments.noisy, "val": kind_count, "test": kind_count}
    generate_dataset(
        arguments.dataset,
        arguments.source,
        arguments.base_seed,
        ComponentTreeKind(arguments.tree),
        tuple(make_noise(name, arguments) for name in arguments.noise),
        noisy_counts,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
