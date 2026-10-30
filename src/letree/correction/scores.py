"""Scores comparing a decoded correction with the clean image."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from letree.generation.targets import CorrectionAction


@dataclass
class CorrectionScores:
    lost_clean_zone_fraction: float
    lost_clean_area_fraction: float
    misassigned_pixel_fraction: float
    depth_mae: float
    exact_depth_fraction: float
    clean_root_lost: bool


def tree_correction(
    parents: NDArray[np.integer],
    levels: NDArray[np.integer],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return corrective parents and actions that decode to this tree's depth."""

    # A level change adds one depth step and an equal-level link adds none.
    # The root compares with itself, so it gets KEEP_ZERO as the decoder requires.
    same_level = levels == levels[parents]
    corrective_actions = np.where(
        same_level, CorrectionAction.KEEP_ZERO, CorrectionAction.KEEP_ONE
    ).astype(np.int64)
    return parents.astype(np.int64), corrective_actions


def score_correction(
    group_map: NDArray[np.integer],
    pixel_depth: NDArray[np.int64],
    clean_label_map: NDArray[np.integer],
    clean_pixel_depth: NDArray[np.int64],
    clean_root: int,
) -> CorrectionScores:
    """Score one decoded hierarchy against the clean image of the same sample."""

    area = group_map.size
    clean_labels = clean_label_map.ravel()
    clean_zone_count = int(clean_labels.max()) + 1

    # Count only the (group, clean zone) pairs that occur.
    pairs, counts = np.unique(
        np.stack([group_map.ravel(), clean_labels], axis=1),
        axis=0,
        return_counts=True,
    )
    groups, zones = pairs[:, 0], pairs[:, 1]

    # Each group is matched to the zone it overlaps most, ties going to the
    # smaller clean id. After this sort that pair comes first in its group.
    order = np.lexsort((zones, -counts, groups))
    groups, zones, counts = groups[order], zones[order], counts[order]
    first = np.ones(groups.size, dtype=bool)
    first[1:] = groups[1:] != groups[:-1]
    best_zones, kept = zones[first], counts[first].sum()

    lost = np.ones(clean_zone_count, dtype=bool)
    lost[best_zones] = False
    lost_area = np.bincount(clean_labels, minlength=clean_zone_count)[lost].sum()

    difference = np.abs(pixel_depth - clean_pixel_depth)

    return CorrectionScores(
        lost_clean_zone_fraction=float(lost.sum() / clean_zone_count),
        lost_clean_area_fraction=float(lost_area / area),
        misassigned_pixel_fraction=float((area - kept) / area),
        depth_mae=float(difference.mean()),
        exact_depth_fraction=float((difference == 0).mean()),
        clean_root_lost=bool(lost[clean_root]),
    )
