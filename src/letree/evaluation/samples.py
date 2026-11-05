"""Chosen samples and the maps their figures show."""

from pathlib import Path

import numpy as np
from torch_geometric.data import Batch

from letree.data.families import load_observation
from letree.data.graphs import to_graph
from letree.evaluation.checkpoint import predict
from letree.evaluation.records import SOURCES, clean_reference, decode_source
from letree.evaluation.scores import pixel_errors


def sample_key(family: str, position: int) -> str:
    # Such as sample_00011681/2, the same for every checkpoint
    return f"{Path(family).stem}/{position}"


def split_samples(dataset_root: str | Path, split: str) -> list[tuple[str, int, str]]:
    """Family file name, position and corruption of every observation."""
    samples = []
    for path in sorted((Path(dataset_root) / split).glob("sample_*.npz")):
        with np.load(path) as arrays:
            noises = arrays["noise"].tolist()
        samples += [
            (path.name, position, noise) for position, noise in enumerate(noises)
        ]
    return samples


def choose_samples(
    samples: list[tuple[str, int, str]], per_group: int, seed: int
) -> list[str]:
    """Keys of per_group samples for clean and each noise kind, one per family."""
    rng = np.random.default_rng(seed)
    # Sorted first, so the draw depends only on the seed
    samples = sorted(samples)
    used_families, keys = set(), []
    for group in sorted({corruption for _, _, corruption in samples}):
        candidates = [sample for sample in samples if sample[2] == group]
        chosen = 0
        for index in rng.permutation(len(candidates)):
            family, position, _ = candidates[index]
            if family in used_families:
                continue
            used_families.add(family)
            keys.append(sample_key(family, position))
            chosen += 1
            if chosen == per_group:
                break
    return keys


def sample_maps(
    dataset_root: str | Path, split: str, keys: list[str], model, settings, device
) -> list[dict]:
    stored = []
    for key in keys:
        family, position = key.split("/")
        path = Path(dataset_root) / split / f"{family}.npz"
        stored.append((key, path, load_observation(path, int(position))))

    data = Batch.from_data_list(
        [
            to_graph(observation.tree_image, observation.targets)
            for _, _, observation in stored
        ]
    )
    predictions = predict(model, settings, data.to(device))

    samples = []
    for (key, path, observation), prediction in zip(stored, predictions):
        clean = load_observation(path, 0)
        clean_labels, clean_depth, _ = clean_reference(clean)
        sample = {
            "key": key,
            "corruption": observation.noise,
            "clean_image": clean.tree_image.image,
            "image": observation.tree_image.image,
            "clean_labels": clean_labels,
            "clean_depth": clean_depth,
            "observed_labels": observation.tree_image.graph.label_map,
        }
        for source in SOURCES:
            counts, hierarchy = decode_source(source, observation, prediction)
            if hierarchy is None:
                sample[source] = counts
                continue
            groups, depth = hierarchy
            errors = pixel_errors(groups, depth, clean_labels, clean_depth)
            sample[source] = counts | {
                "group_map": groups,
                "depth": depth,
                "misassigned": errors.misassigned,
                "depth_difference": errors.depth_difference,
            }
        samples.append(sample)
    return samples
