"""Running a trained checkpoint over a split."""

import json
from dataclasses import replace
from math import sqrt
from pathlib import Path

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from letree.data.graphs import ObservationDataset, make_loader
from letree.evaluation.records import GraphPrediction, sample_records
from letree.evaluation.summary import summarize_records, write_records
from letree.experiments.metrics import graph_local_argmax
from letree.experiments.train import (
    PROGRESS_SECONDS,
    RunSettings,
    build_model,
    run_model,
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
) -> list[GraphPrediction]:
    """Outputs of the last corrective step for every graph of the batch."""
    output = run_model(model, data, settings, settings.correction_steps)
    step = output.corrective[-1]
    pointers, tied = graph_local_argmax(step.query, step.key, data.ptr)
    increment_probabilities = torch.sigmoid(output.increment[-1])
    contraction_logits = output.contraction[-1]
    # Same scale as the losses, so these are the probabilities the model was trained on
    scaled_query = step.query / sqrt(step.query.shape[1])

    # Batching keeps each graph's RAG edges together and in stored order
    edge_counts = torch.bincount(
        data.batch[data.rag_edge_index[0]], minlength=data.num_graphs
    )
    node_bounds = data.ptr.tolist()
    edge_bounds = [0, *torch.cumsum(edge_counts, 0).tolist()]

    predictions = []
    for graph in range(data.num_graphs):
        nodes = slice(node_bounds[graph], node_bounds[graph + 1])
        edges = slice(edge_bounds[graph], edge_bounds[graph + 1])
        scores = scaled_query[nodes] @ step.key[nodes].T
        predictions.append(
            GraphPrediction(
                contraction_logits=contraction_logits[edges].float().cpu().numpy(),
                increment_probabilities=(
                    increment_probabilities[nodes].float().cpu().numpy()
                ),
                pointer_probabilities=torch.softmax(scores.float(), dim=1)
                .cpu()
                .numpy(),
                pointers=pointers[nodes].cpu().numpy(),
                ties=int(tied[nodes].sum()),
            )
        )
    return predictions


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
        predictions = predict(model, settings, data.to(device))
        for item, prediction in zip(data.item.tolist(), predictions):
            path, position = dataset.items[item]
            records += sample_records(path, position, prediction)
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
