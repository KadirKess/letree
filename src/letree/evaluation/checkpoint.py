"""Running a trained checkpoint over a split."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Batch
from tqdm import tqdm

from letree.data.graphs import ObservationDataset, make_loader
from letree.evaluation.records import sample_records
from letree.evaluation.summary import summarize_records, write_records
from letree.experiments.metrics import graph_local_argmax
from letree.experiments.train import (
    PROGRESS_SECONDS,
    RunSettings,
    build_model,
    select_device,
    write_json,
)
from letree.model import LetreeModel


def load_run_model(
    run_directory: str | Path, checkpoint: str = "best.pt", **overrides
) -> tuple[LetreeModel, RunSettings, torch.device]:
    run_directory = Path(run_directory)
    stored = json.loads((run_directory / "settings.json").read_text())["settings"]
    stored["aggregations"] = tuple(stored["aggregations"])
    # Overrides such as attention="dense" keep the computed function
    settings = replace(RunSettings(**stored), **overrides)

    device = select_device()
    if settings.precision == "tf32":
        torch.set_float32_matmul_precision("high")
    model = build_model(settings).to(device)
    state = torch.load(run_directory / checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, settings, device


@torch.no_grad()
def predict(
    model: LetreeModel, settings: RunSettings, data: Batch
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Graph-local corrective parents, actions and tie flags at the last step."""
    output = model(
        data.x,
        data.edge_index,
        data.ptr,
        settings.observed_steps,
        settings.correction_steps,
    )
    step = output.corrective[-1]
    parents, tied = graph_local_argmax(step.query, step.key, data.ptr)
    actions = output.actions[-1].argmax(dim=1)
    return parents.cpu().numpy(), actions.cpu().numpy(), tied.cpu().numpy()


def evaluate_checkpoint(
    run_directory: str | Path,
    dataset_root: str | Path,
    split: str,
    checkpoint: str = "best.pt",
    batch_size: int = 16,
    num_workers: int = 0,
    **overrides,
) -> list[dict]:
    model, settings, device = load_run_model(run_directory, checkpoint, **overrides)
    dataset = ObservationDataset(dataset_root, split, include_noisy=True)
    loader = make_loader(dataset, batch_size, False, settings.seed, num_workers)
    records = []
    for data in tqdm(loader, desc=split, mininterval=PROGRESS_SECONDS):
        parents, actions, tied = predict(model, settings, data.to(device))
        bounds = data.ptr.tolist()
        for index, item in enumerate(data.item.tolist()):
            nodes = slice(bounds[index], bounds[index + 1])
            path, position = dataset.items[item]
            records += sample_records(
                path, position, parents[nodes], actions[nodes], int(tied[nodes].sum())
            )
    return records


def run_evaluation(
    run_directory: str | Path,
    dataset_root: str | Path,
    split: str,
    checkpoint: str = "best.pt",
    batch_size: int = 16,
    num_workers: int = 0,
    **overrides,
) -> dict[str, float]:
    run_directory = Path(run_directory)
    records = evaluate_checkpoint(
        run_directory,
        dataset_root,
        split,
        checkpoint,
        batch_size,
        num_workers,
        **overrides,
    )
    summary = summarize_records(records)
    stem = f"{split}_{Path(checkpoint).stem}"
    write_records(run_directory / f"{stem}_records.csv", records)
    write_json(run_directory / f"{stem}_summary.json", summary)
    return summary
