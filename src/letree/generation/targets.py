from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from letree.component_tree.types import ComponentTree, FlatZoneGraph


@dataclass
class CorrectionTargets:
    # Per observed node, its clean zone, the parent zone of that zone and whether
    # the link to the parent zone adds one level of depth.
    zone: NDArray[np.int64]
    parent_zone: NDArray[np.int64]
    increment: NDArray[np.int64]


@dataclass
class CorrectionDiagnostics:
    """Clean-reference diagnostics for analysis, never model or decoder inputs."""

    # Per observed zone, the number of distinct clean zones beneath its pixels.
    overlap_clean_zone_count: NDArray[np.int64]
    # Per observed zone, the fraction of pixels belonging to its assigned clean role.
    overlap_purity: NDArray[np.float64]
    # Per clean zone, whether no observed zone selected it as a clean role.
    missing_clean_zones: NDArray[np.bool_]
    # Whether no observed zone selected the clean root as its clean role.
    clean_root_missing: bool
    # Former forest roots attached to the selected root, excluding its self-link.
    artificial_root_attachment_count: int


def derive_correction_targets(
    clean_graph: FlatZoneGraph,
    clean_tree: ComponentTree,
    observed_graph: FlatZoneGraph,
) -> tuple[CorrectionTargets, CorrectionDiagnostics]:
    """Derive the zone, parent zone and increment of every observed zone."""
    overlap_counts = _count_zone_overlaps(
        clean_graph.label_map,
        observed_graph.label_map,
        observed_graph.levels.size,
    )
    clean_role, represented = _assign_clean_roles(overlap_counts, clean_graph.anchors)
    targets, forest_roots = _derive_zone_targets(
        clean_role, represented, clean_graph, clean_tree
    )
    diagnostics = _derive_correction_diagnostics(
        overlap_counts,
        clean_role,
        represented,
        clean_tree.parents,
        observed_graph.pixel_counts,
        forest_roots.size,
    )
    return targets, diagnostics


def _count_zone_overlaps(
    clean_label_map: NDArray[np.integer],
    observed_label_map: NDArray[np.integer],
    observed_zone_count: int,
) -> list[dict[int, int]]:
    overlap_counts = [{} for _ in range(observed_zone_count)]

    # Read corresponding pixels once and store only zone pairs that occur.
    for clean_label, observed_label in zip(
        clean_label_map.ravel(), observed_label_map.ravel()
    ):
        clean_zone = int(clean_label)
        observed_zone = int(observed_label)
        counts = overlap_counts[observed_zone]
        counts[clean_zone] = counts.get(clean_zone, 0) + 1

    return overlap_counts


def _assign_clean_roles(
    overlap_counts: list[dict[int, int]],
    clean_anchors: NDArray[np.integer],
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    clean_role = np.full(len(overlap_counts), -1, dtype=np.int64)
    represented = np.zeros(clean_anchors.size, dtype=np.bool_)

    for observed_zone, overlaps in enumerate(overlap_counts):
        max_clean_zone = max(overlaps, key=overlaps.get)
        max_count = overlaps[max_clean_zone]
        tied_clean_zones = [
            clean_zone for clean_zone, count in overlaps.items() if count == max_count
        ]

        # Clean anchors decide equal overlaps independently of zone numbering.
        clean_role[observed_zone] = min(
            tied_clean_zones, key=lambda zone: clean_anchors[zone]
        )

        represented[clean_role[observed_zone]] = True

    return clean_role, represented


def _find_represented_ancestors(
    clean_parents: NDArray[np.integer],
    represented: NDArray[np.bool_],
) -> NDArray[np.int64]:
    children = [[] for _ in range(clean_parents.size)]
    for clean_zone, clean_parent in enumerate(clean_parents):
        if clean_parent == clean_zone:
            root = clean_zone
        else:
            children[clean_parent].append(clean_zone)

    nearest_ancestor = np.full(clean_parents.size, -1, dtype=np.int64)
    pending = [root]

    # Parents are visited first so children can reuse the nearest represented zone.
    # Missing zones pass that ancestor onward. The root has no proper ancestor.
    while pending:
        clean_zone = pending.pop()
        if represented[clean_zone]:
            ancestor_for_children = clean_zone
        else:
            ancestor_for_children = nearest_ancestor[clean_zone]

        for child in children[clean_zone]:
            nearest_ancestor[child] = ancestor_for_children
            pending.append(child)

    return nearest_ancestor


def _derive_zone_targets(
    clean_role: NDArray[np.int64],
    represented: NDArray[np.bool_],
    clean_graph: FlatZoneGraph,
    clean_tree: ComponentTree,
) -> tuple[CorrectionTargets, NDArray[np.int64]]:
    nearest_ancestor = _find_represented_ancestors(clean_tree.parents, represented)

    # Links are built between clean zones and only then read per observed zone,
    # so every node of a zone shares its targets.
    forest_roots = np.flatnonzero(represented & (nearest_ancestor == -1))
    selected_root = _select_correction_root(forest_roots, clean_graph, clean_tree)
    zone_parents = nearest_ancestor
    zone_parents[forest_roots] = selected_root

    # Only represented zones are read, and they all have a parent zone by now.
    parent_zone = zone_parents[clean_role]
    levels = clean_graph.levels
    increment = (levels[clean_role] != levels[parent_zone]).astype(np.int64)

    # Former forest roots stay separate and add no depth beneath the selected root.
    increment[np.isin(clean_role, forest_roots)] = 0

    return CorrectionTargets(clean_role, parent_zone, increment), forest_roots


def _select_correction_root(
    forest_roots: NDArray[np.int64],
    clean_graph: FlatZoneGraph,
    clean_tree: ComponentTree,
) -> int:
    sign = 1 if clean_tree.kind == "max_tree" else -1

    # Max-trees choose the lowest clean level and min-trees the highest. Equal
    # levels go to the larger zone, then the smaller anchor, as canonical zones do.
    def order(zone):
        return (
            sign * int(clean_graph.levels[zone]),
            -int(clean_graph.pixel_counts[zone]),
            int(clean_graph.anchors[zone]),
        )

    return int(min(forest_roots, key=order))


def _derive_correction_diagnostics(
    overlap_counts: list[dict[int, int]],
    clean_role: NDArray[np.int64],
    represented: NDArray[np.bool_],
    clean_parents: NDArray[np.integer],
    observed_pixel_counts: NDArray[np.integer],
    forest_root_count: int,
) -> CorrectionDiagnostics:
    overlap_clean_zone_count = np.empty(clean_role.size, dtype=np.int64)
    overlap_purity = np.empty(clean_role.size, dtype=np.float64)

    # Reuse the counted overlaps instead of reading the paired images again.
    for observed_zone, overlaps in enumerate(overlap_counts):
        overlap_clean_zone_count[observed_zone] = len(overlaps)
        assigned_pixel_count = overlaps[clean_role[observed_zone]]
        overlap_purity[observed_zone] = (
            assigned_pixel_count / observed_pixel_counts[observed_zone]
        )

    missing_clean_zones = ~represented
    clean_root = np.flatnonzero(clean_parents == np.arange(clean_parents.size))[0]
    clean_root_missing = bool(missing_clean_zones[clean_root])

    return CorrectionDiagnostics(
        overlap_clean_zone_count=overlap_clean_zone_count,
        overlap_purity=overlap_purity,
        missing_clean_zones=missing_clean_zones,
        clean_root_missing=clean_root_missing,
        artificial_root_attachment_count=forest_root_count - 1,
    )
