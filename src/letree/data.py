"""Stored observations served as PyG graphs."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from letree.dataset.store import NpzDataset
from letree.generation.noise import NOISE_KIND_NAMES
from letree.generation.samples import GenerationSample


class Graph(Data):
    """One observed flat-zone graph and its targets."""

    def __inc__(self, key, value, *args, **kwargs):
        # Parent targets are local to their graph, so batching must not shift them
        if key in {"observed_parents", "corrective_parents"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def to_graph(sample: GenerationSample) -> Graph:
    """Node features are level, area and anchor, each scaled to [0, 1]."""
    graph = sample.observed.graph
    pixel_count = graph.label_map.size
    features = np.column_stack(
        (
            graph.levels / 255.0,
            graph.pixel_counts / pixel_count,
            graph.anchors / max(pixel_count - 1, 1),
        )
    )
    node_count = graph.levels.size
    edge_index = torch.as_tensor(graph.edge_index.T, dtype=torch.int64)
    edge_index = to_undirected(edge_index, num_nodes=node_count)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=node_count)

    return Graph(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        observed_parents=torch.as_tensor(
            sample.observed.component_tree.parents, dtype=torch.int64
        ),
        corrective_parents=torch.as_tensor(
            sample.correction_targets.corrective_parents, dtype=torch.int64
        ),
        correction_actions=torch.as_tensor(
            sample.correction_targets.actions, dtype=torch.int64
        ),
        num_nodes=node_count,
    )


class ObservationDataset(Dataset):
    """One item per observation, loading its family file lazily.

    Position zero is the clean observation of a family, and positions from one
    onward are its noisy observations in stored order.
    """

    def __init__(self, root: str | Path, split: str, include_noisy: bool) -> None:
        self.dataset = NpzDataset.open(root)
        self.split = split
        sample_indices = sorted(
            int(path.stem.removeprefix("sample_"))
            for path in (Path(root) / split).glob("sample_*.npz")
        )
        if not sample_indices:
            raise ValueError(f"Split {split!r} of {root} contains no families.")

        positions = [0]
        if include_noisy:
            noisy_count = len(self.dataset.load(split, sample_indices[0]).noisy)
            positions += range(1, noisy_count + 1)
        self.items = [
            (index, position) for index in sample_indices for position in positions
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Graph:
        sample_index, position = self.items[item]
        family = self.dataset.load(self.split, sample_index)
        if position == 0:
            sample, corruption = family.clean, "clean"
        else:
            observation = family.noisy[position - 1]
            sample = observation.sample
            corruption = NOISE_KIND_NAMES[type(observation.noise_config)]

        graph = to_graph(sample)
        graph.is_clean = torch.tensor(position == 0)
        graph.corruption = corruption
        return graph


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
