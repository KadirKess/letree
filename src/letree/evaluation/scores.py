"""Scores comparing a decoded correction with the clean image."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CorrectionScores:
    lost_clean_zone_fraction: float
    lost_clean_area_fraction: float
    misassigned_pixel_fraction: float
    depth_mae: float
    exact_depth_fraction: float
    clean_root_lost: bool


@dataclass
class PixelErrors:
    # Per pixel, whether its group is matched to another clean zone than its own
    misassigned: NDArray[np.bool_]
    depth_difference: NDArray[np.int64]
    # Per clean zone, whether no decoded group is matched to it
    lost_clean_zones: NDArray[np.bool_]


def pixel_errors(
    group_map: NDArray[np.integer],
    pixel_depth: NDArray[np.int64],
    clean_label_map: NDArray[np.integer],
    clean_pixel_depth: NDArray[np.int64],
) -> PixelErrors:
    """Errors of one decoded hierarchy per pixel and per clean zone."""

    groups = group_map.ravel()
    clean_labels = clean_label_map.ravel()
    clean_zone_count = int(clean_labels.max()) + 1

    # Count only the (group, clean zone) pairs that occur.
    pairs, counts = np.unique(
        np.stack([groups, clean_labels], axis=1), axis=0, return_counts=True
    )
    pair_groups, pair_zones = pairs[:, 0], pairs[:, 1]

    # Each group is matched to the zone it overlaps most, ties going to the
    # smaller clean id. After this sort that pair comes first in its group.
    order = np.lexsort((pair_zones, -counts, pair_groups))
    pair_groups, pair_zones = pair_groups[order], pair_zones[order]
    first = np.ones(pair_groups.size, dtype=bool)
    first[1:] = pair_groups[1:] != pair_groups[:-1]

    matched_zone = np.empty(int(groups.max()) + 1, dtype=np.int64)
    matched_zone[pair_groups[first]] = pair_zones[first]

    lost = np.ones(clean_zone_count, dtype=bool)
    lost[pair_zones[first]] = False

    return PixelErrors(
        misassigned=matched_zone[group_map] != clean_label_map,
        depth_difference=np.abs(pixel_depth - clean_pixel_depth),
        lost_clean_zones=lost,
    )


def score_correction(
    group_map: NDArray[np.integer],
    pixel_depth: NDArray[np.int64],
    clean_label_map: NDArray[np.integer],
    clean_pixel_depth: NDArray[np.int64],
    clean_root: int,
) -> CorrectionScores:
    """Score one decoded hierarchy against the clean image of the same sample."""

    errors = pixel_errors(group_map, pixel_depth, clean_label_map, clean_pixel_depth)
    lost = errors.lost_clean_zones
    clean_area = np.bincount(clean_label_map.ravel(), minlength=lost.size)

    return CorrectionScores(
        lost_clean_zone_fraction=float(lost.mean()),
        lost_clean_area_fraction=float(clean_area[lost].sum() / group_map.size),
        misassigned_pixel_fraction=float(errors.misassigned.mean()),
        depth_mae=float(errors.depth_difference.mean()),
        exact_depth_fraction=float((errors.depth_difference == 0).mean()),
        clean_root_lost=bool(lost[clean_root]),
    )
