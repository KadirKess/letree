"""A training run: its settings, epochs, checkpoint selection and test result."""

import json
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import inf
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from letree.data import ObservationDataset, make_loader
from letree.losses import correction_losses
from letree.metrics import batch_image_metrics, summarize_image_metrics
from letree.model import LetreeModel

SELECTION_METRIC = "all/joint_correction_accuracy/image_mean"

# Colab redraws subprocess output slowly, so refresh the bars at most every few seconds
PROGRESS_SECONDS = 5


@dataclass(frozen=True)
class RunSettings:
    dataset_root: str
    max_epochs: int
    include_noisy: bool = True
    width: int = 128
    correction_recurrent: bool = True
    aggregations: tuple[str, ...] = ("max", "min")
    attention: str = "dense"
    observed_steps: int = 10
    correction_steps: int = 5
    plateau_patience: int = 3
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    plateau_factor: float = 0.1
    gradient_clip_norm: float = 1.0
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 0


def build_model(settings: RunSettings) -> LetreeModel:
    return LetreeModel(
        settings.width,
        settings.correction_recurrent,
        settings.aggregations,
        settings.attention,
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, settings, device) -> dict[str, float]:
    """Mean training losses over the epoch's images."""
    model.train()
    totals = defaultdict(float)
    graph_count = 0
    progress = tqdm(loader, desc="train", leave=False, mininterval=PROGRESS_SECONDS)
    for data in progress:
        data = data.to(device)
        output = model(
            data.x,
            data.edge_index,
            data.ptr,
            settings.observed_steps,
            settings.correction_steps,
        )
        losses = correction_losses(output, data)

        optimizer.zero_grad()
        sum(losses.values()).backward()
        clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()

        # Weight each batch by its graphs, since the last batch can be smaller
        for name, value in losses.items():
            totals[name] += value.item() * data.num_graphs
        graph_count += data.num_graphs
        progress.set_postfix(
            loss=f"{sum(totals.values()) / graph_count:.4f}", refresh=False
        )

    return {name: total / graph_count for name, total in totals.items()}


@torch.no_grad()
def evaluate(model, loader, settings, device) -> dict[str, float]:
    """Split losses, accuracies, baselines and tie counts."""
    model.eval()
    per_image = defaultdict(list)
    loss_totals = defaultdict(float)
    graph_count = 0
    for data in tqdm(
        loader, desc="evaluate", leave=False, mininterval=PROGRESS_SECONDS
    ):
        data = data.to(device)
        output = model(
            data.x,
            data.edge_index,
            data.ptr,
            settings.observed_steps,
            settings.correction_steps,
        )
        for name, value in correction_losses(output, data).items():
            loss_totals[name] += value.item() * data.num_graphs
        graph_count += data.num_graphs

        for name, value in batch_image_metrics(output, data).items():
            per_image[name].append(value.cpu())

    summary = summarize_image_metrics(per_image)
    for name, total in loss_totals.items():
        summary[f"loss/{name}"] = total / graph_count
    return summary


LOSS_NAMES = ("observed_parent", "corrective_parent", "action")
ACCURACY_NAMES = (
    "observed_parent_accuracy",
    "corrective_parent_accuracy",
    "action_accuracy",
    "joint_correction_accuracy",
)


def epoch_report(history: list[dict], stale: int, patience: int) -> str:
    """Console summary of the latest epoch, one row per loss or accuracy source."""
    entry = history[-1]
    train, validation = entry["train"], entry["validation"]
    # The best epoch is stale epochs ago, or none when no score was finite
    best = history[-1 - stale] if stale < len(history) else None

    def row(label: str, values: list[float]) -> str:
        return f"  {label:<24}" + "".join(f"{value:>12.4f}" for value in values)

    validation_losses = [validation[f"loss/{n}"] for n in LOSS_NAMES]
    lines = [
        f"epoch {entry['epoch']}  lr {entry['learning_rate']:.1e}  "
        f"{entry['seconds']:.0f}s",
        f"  {'loss':<24}"
        + "".join(f"{name:>12}" for name in ("observed", "corrective", "action"))
        + f"{'total':>12}",
        row(
            "train (epoch mean)", [train[n] for n in LOSS_NAMES] + [sum(train.values())]
        ),
        row("val", validation_losses + [sum(validation_losses)]),
        f"  {'accuracy, image mean':<24}"
        + "".join(
            f"{name:>12}" for name in ("observed", "corrective", "action", "joint")
        ),
    ]
    for group in ("clean", "noisy"):
        if f"{group}/images" in validation:
            lines.append(
                row(
                    f"val {group}",
                    [validation[f"{group}/{n}/image_mean"] for n in ACCURACY_NAMES],
                )
            )
    best_text = (
        "none yet"
        if best is None
        else f"{best['validation'][SELECTION_METRIC]:.4f} at epoch {best['epoch']}"
    )
    lines.append(
        f"  selection (val joint, image mean) {validation[SELECTION_METRIC]:.4f}  "
        f"best {best_text}  stale {stale}/{patience}"
    )
    return "\n".join(lines)


def save_checkpoint(path: Path, state: dict) -> None:
    # Write then rename, so a disconnect never leaves a truncated checkpoint
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_json(path: Path, content: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2))
    temporary.replace(path)


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit(
    settings: RunSettings, run_directory: str | Path, resume: bool = False
) -> dict[str, float]:
    """Train, keep the best validation checkpoint and evaluate test once."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    last_path = run_directory / "last.pt"
    best_path = run_directory / "best.pt"

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    device = select_device()

    def loader(split: str, shuffle: bool):
        dataset = ObservationDataset(
            settings.dataset_root, split, settings.include_noisy
        )
        return make_loader(
            dataset, settings.batch_size, shuffle, settings.seed, settings.num_workers
        )

    train_loader = loader("train", shuffle=True)
    validation_loader = loader("val", shuffle=False)

    model = build_model(settings).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=settings.plateau_factor,
        patience=settings.plateau_patience,
    )

    first_epoch, best, stale, history = 0, -inf, 0, []
    if resume and last_path.exists():
        # A resumed run continues the schedule, but its shuffling order differs
        # from an uninterrupted run
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch = state["epoch"] + 1
        best, stale, history = state["best"], state["stale"], state["history"]
    else:
        manifest = Path(settings.dataset_root) / "manifest.json"
        write_json(
            run_directory / "settings.json",
            {
                "settings": asdict(settings),
                "selection_metric": SELECTION_METRIC,
                "git": git_revision(),
                "torch": torch.__version__,
                "device": str(device),
                "dataset_manifest": json.loads(manifest.read_text()),
            },
        )

    for epoch in range(first_epoch, settings.max_epochs):
        if stale >= settings.early_stopping_patience:
            break

        started = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, settings, device)
        validation = evaluate(model, validation_loader, settings, device)
        score = validation[SELECTION_METRIC]
        scheduler.step(score)

        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append(
            {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train,
                "validation": validation,
            }
        )

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "history": history,
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        write_json(run_directory / "history.json", history)

        mean_seconds = sum(entry["seconds"] for entry in history) / len(history)
        remaining_hours = mean_seconds * (settings.max_epochs - epoch - 1) / 3600
        print(
            epoch_report(history, stale, settings.early_stopping_patience)
            + f"\n  at most {remaining_hours:.1f}h left\n",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError("No epoch produced a finite validation score.")
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test = evaluate(model, loader("test", shuffle=False), settings, device)
    write_json(run_directory / "test.json", test)
    return test
