"""Per-sample score records for the model, the observed tree and the oracle."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from letree.correction.decoder import CorrectionDecodeError, decode_correction
from letree.data.families import Observation, load_observation
from letree.evaluation.scores import pixel_errors, tree_correction

SOURCES = ("model", "observed", "oracle")


def clean_reference(
    clean: Observation,
) -> tuple[NDArray[np.integer], NDArray[np.int64], int]:
    """Clean label map, clean pixel depth and clean root of a family."""
    graph = clean.tree_image.graph
    parents = clean.tree_image.component_tree.parents
    decoded = decode_correction(
        *tree_correction(parents, graph.levels), graph.label_map
    )
    root = int(np.flatnonzero(parents == np.arange(parents.size))[0])
    return graph.label_map, decoded.pixel_depth, root


def source_predictions(
    observation: Observation,
    model_parents: NDArray[np.integer],
    model_actions: NDArray[np.integer],
) -> dict[str, tuple[NDArray[np.integer], NDArray[np.integer]]]:
    tree = observation.tree_image
    return {
        "model": (model_parents, model_actions),
        "observed": tree_correction(tree.component_tree.parents, tree.graph.levels),
        "oracle": (
            observation.targets.corrective_parents,
            observation.targets.actions,
        ),
    }


def score_record(
    corrective_parents: NDArray[np.integer],
    actions: NDArray[np.integer],
    observation: Observation,
    reference: tuple[NDArray[np.integer], NDArray[np.int64], int],
) -> dict:
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

    try:
        decoded = decode_correction(corrective_parents, actions, label_map)
    except CorrectionDecodeError as error:
        return record | {"valid": False, "failure": str(error.reason), "exact": False}

    errors = pixel_errors(
        decoded.observed_to_output[label_map],
        decoded.pixel_depth,
        clean_labels,
        clean_depth,
    )
    lost = errors.lost_clean_zones
    clean_area = np.bincount(clean_labels.ravel(), minlength=lost.size)
    record |= {
        "valid": True,
        "failure": "",
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


def sample_records(
    path: Path,
    position: int,
    model_parents: NDArray[np.integer],
    model_actions: NDArray[np.integer],
    model_ties: int,
) -> list[dict]:
    observation = load_observation(path, position)
    reference = clean_reference(load_observation(path, 0))
    sample = {
        "family": path.name,
        "position": position,
        "corruption": observation.noise,
        "nodes": int(observation.tree_image.graph.levels.size),
    }
    predictions = source_predictions(observation, model_parents, model_actions)
    return [
        sample
        | {"source": source, "ties": model_ties if source == "model" else 0}
        | score_record(parents, actions, observation, reference)
        for source, (parents, actions) in predictions.items()
    ]
