"""Stored observations served as PyG graphs."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from letree.data.families import load_observation
from letree.generation.samples import TreeImage
from letree.generation.targets import CorrectionTargets


class Graph(Data):
    """One observed flat-zone graph and its targets."""

    def __inc__(self, key, value, *args, **kwargs):
        # Parent targets are local to their graph, so batching must not shift them
        if key in {"observed_parents", "corrective_parents"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def to_graph(tree_image: TreeImage, targets: CorrectionTargets) -> Graph:
    """Node features are level, area and anchor, each scaled to [0, 1]."""
    graph = tree_image.graph
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
            tree_image.component_tree.parents, dtype=torch.int64
        ),
        corrective_parents=torch.as_tensor(
            targets.corrective_parents, dtype=torch.int64
        ),
        correction_actions=torch.as_tensor(targets.actions, dtype=torch.int64),
        num_nodes=node_count,
    )


class ObservationDataset(Dataset):
    """One item per observation, reading only that observation from its family file.

    Position zero is the clean observation of a family, and positions from one
    onward are its noisy observations in stored order.
    """

    def __init__(self, root: str | Path, split: str, include_noisy: bool) -> None:
        paths = sorted((Path(root) / split).glob("sample_*.npz"))
        if not paths:
            raise ValueError(f"Split {split!r} of {root} contains no families.")
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        noisy_count = manifest["noisy_counts"][split]
        positions = range(1 + noisy_count) if include_noisy else [0]
        self.items = [(path, position) for path in paths for position in positions]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Graph:
        observation = load_observation(*self.items[item])
        graph = to_graph(observation.tree_image, observation.targets)
        graph.corruption = observation.noise
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
