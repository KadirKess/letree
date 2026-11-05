import tracemalloc

import numpy as np
import pytest

from letree.component_tree.reference import (
    build_reference_component_tree,
    build_reference_rag,
)
from letree.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from letree.generation import derive_correction_targets


def _targets_for(clean, observed, kind):
    clean_image = np.array(clean, dtype=np.uint8)
    observed_image = np.array(observed, dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)
    return derive_correction_targets(clean_graph, clean_tree, observed_graph)


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "clean, observed, zone, parent_zone, increment",
    [
        ([10], [10], [0], [0], [0]),
        ([10, 20, 10], [10, 20, 10], [0, 1, 2], [0, 0, 0], [0, 1, 0]),
        ([30, 20, 10], [30, 20, 10], [0, 1, 2], [1, 2, 2], [1, 1, 0]),
        ([20, 20, 20], [10, 30, 10], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        ([10, 10, 20], [1, 2, 3], [0, 0, 1], [0, 0, 0], [0, 0, 1]),
        ([10, 20, 30], [30, 20, 10], [0, 1, 2], [0, 0, 1], [0, 1, 1]),
        ([10, 20, 30], [7, 7, 8], [0, 2], [0, 0], [0, 1]),
        ([20, 10, 30], [7, 7, 8], [0, 2], [0, 0], [0, 0]),
        ([10, 20, 20], [7, 7, 7], [1], [1], [0]),
    ],
    ids=[
        "one-pixel",
        "equal-level-canonical-link",
        "root-at-last-zone",
        "fragments-with-repeated-observed-level",
        "different-clean-and-observed-ids",
        "branch-reversal",
        "missing-ancestor",
        "missing-root-with-unequal-levels",
        "only-nonroot-role-survives",
    ],
)
def test_named_correction_targets(kind, clean, observed, zone, parent_zone, increment):
    targets, _ = _targets_for([clean], [observed], kind)

    np.testing.assert_array_equal(targets.zone, zone)
    np.testing.assert_array_equal(targets.parent_zone, parent_zone)
    np.testing.assert_array_equal(targets.increment, increment)
    assert targets.zone.dtype == np.int64
    assert targets.parent_zone.dtype == np.int64
    assert targets.increment.dtype == np.int64


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_plan_worked_example(kind):
    # Clean zones A, B and C have ids 0, 1 and 2. A and C share the root component.
    targets, diagnostics = _targets_for(
        [[0, 0, 5, 5, 0, 0]], [[0, 1, 5, 4, 0, 0]], kind
    )

    np.testing.assert_array_equal(targets.zone, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(targets.parent_zone, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(targets.increment, [0, 0, 1, 1, 0])
    assert diagnostics.artificial_root_attachment_count == 0


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_lost_root_keeps_extreme_forest_root(kind):
    # Clean zones are 0 at level 0, 1 at 3, 2 at 0 and 3 at 5. Zones 0 and 2 are lost.
    clean = [[0, 3, 3, 3, 0, 5, 5, 5]]
    observed = [[3, 3, 3, 3, 5, 5, 5, 5]]
    targets, diagnostics = _targets_for(clean, observed, kind)

    # The level 3 zone is closer to the lost root. The other root adds no depth.
    np.testing.assert_array_equal(targets.zone, [1, 3])
    np.testing.assert_array_equal(targets.parent_zone, [1, 1])
    np.testing.assert_array_equal(targets.increment, [0, 0])
    assert diagnostics.clean_root_missing is True
    assert diagnostics.artificial_root_attachment_count == 1


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_lost_root_prefers_larger_forest_root(kind):
    # Clean zones 1 and 3 share level 5, with 2 and 3 pixels. Zones 0 and 2 are lost.
    clean = [[0, 5, 5, 0, 5, 5, 5]]
    observed = [[6, 6, 6, 7, 7, 7, 7]]
    targets, diagnostics = _targets_for(clean, observed, kind)

    # Size decides before the anchor, so the later and larger zone 3 is the root.
    np.testing.assert_array_equal(targets.zone, [1, 3])
    np.testing.assert_array_equal(targets.parent_zone, [3, 3])
    np.testing.assert_array_equal(targets.increment, [0, 0])
    assert diagnostics.clean_root_missing is True
    assert diagnostics.artificial_root_attachment_count == 1


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_split_zone_shares_targets(kind):
    # Clean zone 0 is the top row at 20 and zone 1 the rest at 10.
    clean = [[20, 20, 20], [10, 10, 10], [10, 10, 10]]
    # The level 5 zone cuts the top row, so its two ends are not adjacent.
    observed = [[21, 5, 22], [5, 5, 5], [5, 5, 5]]
    targets, _ = _targets_for(clean, observed, kind)

    np.testing.assert_array_equal(targets.zone, [0, 1, 0])
    np.testing.assert_array_equal(targets.parent_zone, [1, 1, 1])
    np.testing.assert_array_equal(targets.increment, [1, 0, 1])


def _permute_graph(graph, permutation):
    original_ids = np.argsort(permutation)
    return FlatZoneGraph(
        label_map=permutation[graph.label_map],
        edge_index=permutation[graph.edge_index],
        levels=graph.levels[original_ids],
        pixel_counts=graph.pixel_counts[original_ids],
        anchors=graph.anchors[original_ids],
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize("permute_clean", [False, True])
@pytest.mark.parametrize("permute_observed", [False, True])
def test_overlap_and_root_ties_follow_anchors(kind, permute_clean, permute_observed):
    clean_image = np.array([[20, 10, 20]], dtype=np.uint8)
    observed_image = np.array([[7, 7, 8]], dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    clean_permutation = np.array([2, 1, 0]) if permute_clean else np.arange(3)
    observed_permutation = np.array([1, 0]) if permute_observed else np.arange(2)
    clean_graph = _permute_graph(clean_graph, clean_permutation)
    observed_graph = _permute_graph(observed_graph, observed_permutation)
    clean_tree = ComponentTree(
        kind=kind,
        parents=clean_permutation[clean_tree.parents[np.argsort(clean_permutation)]],
    )

    targets, diagnostics = derive_correction_targets(
        clean_graph, clean_tree, observed_graph
    )

    # The overlap tie loses the clean root. The forest roots tie on level and size,
    # so the first clean anchor wins.
    np.testing.assert_array_equal(
        targets.zone[observed_permutation], clean_permutation[[0, 2]]
    )
    np.testing.assert_array_equal(
        targets.parent_zone[observed_permutation], clean_permutation[[0, 0]]
    )
    np.testing.assert_array_equal(targets.increment[observed_permutation], [0, 0])
    np.testing.assert_array_equal(
        diagnostics.overlap_clean_zone_count[observed_permutation], [2, 1]
    )
    np.testing.assert_allclose(
        diagnostics.overlap_purity[observed_permutation], [0.5, 1.0]
    )
    np.testing.assert_array_equal(
        diagnostics.missing_clean_zones[clean_permutation], [False, True, False]
    )
    assert diagnostics.clean_root_missing is True
    assert diagnostics.artificial_root_attachment_count == 1


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "clean, observed, overlap_count, purity, missing, root_missing, attachment_count",
    [
        (
            [10, 20, 10],
            [10, 20, 10],
            [1, 1, 1],
            [1, 1, 1],
            [False, False, False],
            False,
            0,
        ),
        ([20, 20, 20], [10, 30, 10], [1, 1, 1], [1, 1, 1], [False], False, 0),
        (
            [10, 10, 20, 30],
            [7, 7, 7, 8],
            [2, 1],
            [2 / 3, 1],
            [False, True, False],
            False,
            0,
        ),
        (
            [20, 10, 30],
            [7, 7, 8],
            [2, 1],
            [0.5, 1],
            [False, True, False],
            True,
            1,
        ),
        ([10, 20, 20], [7, 7, 7], [2], [2 / 3], [True, False], True, 0),
        (
            [30, 20, 10],
            [7, 8, 8],
            [1, 2],
            [1, 0.5],
            [False, False, True],
            True,
            0,
        ),
        (
            [20, 10, 30, 10, 40],
            [7, 7, 8, 8, 9],
            [2, 2, 1],
            [0.5, 0.5, 1],
            [False, True, False, True, False],
            True,
            2,
        ),
    ],
    ids=[
        "clean-identity",
        "pure-fragments",
        "unequal-mixed-overlap",
        "missing-root-with-two-forest-roots",
        "missing-root-with-one-observed-zone",
        "missing-last-clean-root-with-one-forest-root",
        "three-forest-roots",
    ],
)
def test_named_correction_diagnostics(
    kind,
    clean,
    observed,
    overlap_count,
    purity,
    missing,
    root_missing,
    attachment_count,
):
    clean_image = np.array([clean], dtype=np.uint8)
    observed_image = np.array([observed], dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    _, diagnostics = derive_correction_targets(clean_graph, clean_tree, observed_graph)

    np.testing.assert_array_equal(diagnostics.overlap_clean_zone_count, overlap_count)
    np.testing.assert_allclose(diagnostics.overlap_purity, purity)
    np.testing.assert_array_equal(diagnostics.missing_clean_zones, missing)
    assert diagnostics.clean_root_missing is root_missing
    assert diagnostics.artificial_root_attachment_count == attachment_count
    assert diagnostics.overlap_clean_zone_count.dtype == np.int64
    assert diagnostics.overlap_purity.dtype == np.float64
    assert diagnostics.missing_clean_zones.dtype == np.bool_


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_long_missing_ancestor_chain(kind):
    clean_image = np.arange(256, dtype=np.uint8).reshape(1, 256)
    observed_image = np.zeros((1, 256), dtype=np.uint8)
    observed_image[0, 254:] = [1, 2]
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)

    # The first zone selects role zero. Roles 254 and 255 keep one increment each.
    np.testing.assert_array_equal(targets.zone, [0, 254, 255])
    np.testing.assert_array_equal(targets.parent_zone, [0, 0, 254])
    np.testing.assert_array_equal(targets.increment, [0, 1, 1])


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_checkerboard_identity_at_full_image_scale(kind):
    rows, columns = np.indices((256, 256))
    image = ((rows + columns) % 2).astype(np.uint8)
    graph = build_reference_rag(image)
    tree = build_reference_component_tree(graph, kind)

    assert graph.levels.size == 65_536
    tracemalloc.start()
    try:
        targets, _ = derive_correction_targets(graph, tree, graph)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Sparse construction fits comfortably here. A Cartesian matrix needs gigabytes.
    assert peak_bytes < 64 * 1024**2

    np.testing.assert_array_equal(targets.zone, np.arange(graph.levels.size))
    np.testing.assert_array_equal(targets.parent_zone, tree.parents)
    expected_increment = graph.levels != graph.levels[tree.parents]
    np.testing.assert_array_equal(targets.increment, expected_increment)
