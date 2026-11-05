"""Per-sample score records for both model votes, the observed tree and the oracle."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from letree.component_tree.types import FlatZoneGraph
from letree.correction.decoder import (
    CorrectionDecodeError,
    contraction_groups,
    decode_groups,
    group_increments,
    hard_vote,
    soft_vote,
    tree_pixel_depth,
)
from letree.data.families import Observation, load_observation
from letree.evaluation.scores import pixel_errors
from letree.generation.targets import CorrectionTargets

SOURCES = ("model_soft", "model_hard", "observed", "oracle")


@dataclass
class GraphPrediction:
    # Ids are local to the graph, and contraction follows the stored RAG edge order
    contraction_logits: NDArray[np.floating]
    increment_probabilities: NDArray[np.floating]
    pointer_probabilities: NDArray[np.floating]
    pointers: NDArray[np.int64]
    ties: int


def clean_reference(
    clean: Observation,
) -> tuple[NDArray[np.integer], NDArray[np.int64], int]:
    """Clean label map, clean pixel depth and clean root of a family."""
    graph = clean.tree_image.graph
    parents = clean.tree_image.component_tree.parents
    pixel_depth = tree_pixel_depth(parents, graph.levels, graph.label_map)
    root = int(np.flatnonzero(parents == np.arange(parents.size))[0])
    return graph.label_map, pixel_depth, root


def oracle_pointers(
    zone: NDArray[np.int64], parent_zone: NDArray[np.int64]
) -> NDArray[np.int64]:
    # Any node of the parent zone gives the same tree, so take the smallest
    node_count = zone.size
    first_node = np.full(zone.max() + 1, node_count, dtype=np.int64)
    np.minimum.at(first_node, zone, np.arange(node_count))
    return first_node[parent_zone]


def oracle_votes(
    graph: FlatZoneGraph, targets: CorrectionTargets
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Groups, votes and group increments read from the targets."""
    rag_edges = graph.edge_index.T
    zone = targets.zone
    contracted = zone[rag_edges[0]] == zone[rag_edges[1]]
    groups = contraction_groups(rag_edges, contracted, zone.size)
    votes = hard_vote(oracle_pointers(zone, targets.parent_zone), groups)
    increments = group_increments(targets.increment.astype(np.float64), groups)
    return groups, votes, increments


def source_votes(
    source: str, observation: Observation, prediction: GraphPrediction | None
) -> tuple[NDArray[np.int64], NDArray[np.number], NDArray[np.int64]]:
    """Groups, vote matrix and group increments of a source that is decoded."""
    graph = observation.tree_image.graph
    if source == "oracle":
        return oracle_votes(graph, observation.targets)

    groups = contraction_groups(
        graph.edge_index.T, prediction.contraction_logits > 0, graph.levels.size
    )
    probabilities = prediction.increment_probabilities
    if source == "model_soft":
        votes = soft_vote(prediction.pointer_probabilities, groups)
        increments = group_increments(probabilities, groups)
    else:
        votes = hard_vote(prediction.pointers, groups)
        increments = group_increments((probabilities > 0.5).astype(np.float64), groups)
    return groups, votes, increments


def decode_source(
    source: str, observation: Observation, prediction: GraphPrediction | None
) -> tuple[dict, tuple[NDArray[np.int64], NDArray[np.int64]] | None]:
    """Group and root claim counts, then pixel groups and depths or None on failure."""
    tree = observation.tree_image
    label_map = tree.graph.label_map
    if source == "observed":
        # Every observed zone stays its own group
        depth = tree_pixel_depth(
            tree.component_tree.parents, tree.graph.levels, label_map
        )
        counts = {"groups": int(tree.graph.levels.size), "root_claims": 1}
        return counts | {"failure": ""}, (label_map, depth)

    groups, votes, increments = source_votes(source, observation, prediction)
    group_ids = np.arange(votes.shape[0])
    counts = {
        "groups": int(group_ids.size),
        "root_claims": int((votes.argmax(axis=1) == group_ids).sum()),
    }
    try:
        decoded = decode_groups(groups, votes, increments, label_map)
    except CorrectionDecodeError as error:
        return counts | {"failure": str(error.reason)}, None
    return counts | {"failure": ""}, (groups[label_map], decoded.pixel_depth)


def score_record(
    hierarchy: tuple[NDArray[np.integer], NDArray[np.int64]] | None,
    observation: Observation,
    reference: tuple[NDArray[np.integer], NDArray[np.int64], int],
) -> dict:
    """Scores of one decoded hierarchy, or only its strata when decoding failed."""
    clean_labels, clean_depth, clean_root = reference
    label_map = observation.tree_image.graph.label_map

    # A pixel is pure when its observed zone lies inside one clean zone
    pure = observation.diagnostics.overlap_clean_zone_count[label_map] == 1
    pixel_sets = {"all": np.ones_like(pure), "pure": pure, "mixed": ~pure}
    record = {
        "refining": bool(pure.all()),
        "clean_zones": int(clean_labels.max()) + 1,
        **{f"{name}/pixels": int(mask.sum()) for name, mask in pixel_sets.items()},
    }
    if hierarchy is None:
        return record | {"valid": False, "exact": False}

    errors = pixel_errors(*hierarchy, clean_labels, clean_depth)
    lost = errors.lost_clean_zones
    clean_area = np.bincount(clean_labels.ravel(), minlength=lost.size)
    record |= {
        "valid": True,
        "lost_zones": int(lost.sum()),
        "lost_area": int(clean_area[lost].sum()),
        "root_lost": bool(lost[clean_root]),
    }
    for name, mask in pixel_sets.items():
        difference = errors.depth_difference[mask]
        record[f"{name}/misassigned"] = int(errors.misassigned[mask].sum())
        record[f"{name}/depth_error"] = int(difference.sum())
        record[f"{name}/exact_depth"] = int((difference == 0).sum())

    record["exact"] = (
        record["lost_zones"] == 0
        and record["all/misassigned"] == 0
        and record["all/exact_depth"] == record["all/pixels"]
    )
    return record


def sample_records(path: Path, position: int, prediction: GraphPrediction) -> list:
    observation = load_observation(path, position)
    reference = clean_reference(load_observation(path, 0))
    sample = {
        "family": path.name,
        "position": position,
        "corruption": observation.noise,
        "nodes": int(observation.tree_image.graph.levels.size),
    }
    records = []
    for source in SOURCES:
        counts, hierarchy = decode_source(source, observation, prediction)
        # Only the hard vote reads the argmax pointers, so only it has ties
        ties = prediction.ties if source == "model_hard" else 0
        records.append(
            sample
            | {"source": source, "ties": ties}
            | counts
            | score_record(hierarchy, observation, reference)
        )
    return records
