"""Write a clean source of generated square images.

uv run python scripts/make_squares.py SOURCE
"""

import argparse
from dataclasses import fields

from letree.data.sources import write_square_source
from letree.generation.images import SquareImageConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=1_000)
    parser.add_argument("--test", type=int, default=1_000)
    # Every recipe field can be overridden, for example --max-side 16
    for field in fields(SquareImageConfig):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=int, default=field.default)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = SquareImageConfig(
        **{
            field.name: getattr(arguments, field.name)
            for field in fields(SquareImageConfig)
        }
    )
    write_square_source(
        arguments.source,
        arguments.base_seed,
        {"train": arguments.train, "val": arguments.val, "test": arguments.test},
        config,
    )


if __name__ == "__main__":
    main()
