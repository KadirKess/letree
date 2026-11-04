"""Plot a run's training curves and a few chosen test samples.

uv run python scripts/plot_run.py RUN_DIRECTORY DATASET --per-group 2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from letree.evaluation.checkpoint import load_run_model
from letree.evaluation.figures import sample_figure, training_figure
from letree.evaluation.samples import (
    choose_samples,
    sample_maps,
    split_samples,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=("dense", "flex"))
    parser.add_argument("--compile", choices=("none", "model", "step"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run = arguments.run_directory
    figures = run / "figures"
    (figures / "samples").mkdir(parents=True, exist_ok=True)

    history = json.loads((run / "history.json").read_text())
    training_figure(history).savefig(figures / "training.png", dpi=120)

    # Kept in the run, so every later plot shows the same samples
    keys_path = run / f"{arguments.split}_samples.json"
    if keys_path.exists():
        keys = json.loads(keys_path.read_text())
    else:
        samples = split_samples(arguments.dataset, arguments.split)
        keys = choose_samples(samples, arguments.per_group, arguments.seed)
        keys_path.write_text(json.dumps(keys, indent=1))

    overrides = {
        name: value
        for name, value in (
            ("attention", arguments.attention),
            ("compile", arguments.compile),
        )
        if value is not None
    }
    model, settings, device = load_run_model(run, arguments.checkpoint, **overrides)
    stem = Path(arguments.checkpoint).stem
    for sample in sample_maps(
        arguments.dataset, arguments.split, keys, model, settings, device
    ):
        figure = sample_figure(sample)
        name = sample["key"].replace("/", "_")
        figure.savefig(figures / "samples" / f"{stem}_{name}.png", dpi=110)
        plt.close(figure)
        print(sample["key"], sample["corruption"])


if __name__ == "__main__":
    main()
