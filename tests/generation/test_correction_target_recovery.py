from itertools import product

import numpy as np
import pytest

from letree.component_tree.reference import (
    build_reference_component_tree,
    build_reference_rag,
)
from letree.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from letree.generation import derive_correction_targets
from letree.generation.samples import _build_production_tree


def _checked_graph(image, kind):
    graph, tree = _build_production_tree(image, kind)
    reference = build_reference_rag(image)
    reference_tree = build_reference_component_tree(reference, kind)
    for field in ("label_map", "edge_index", "levels", "pixel_counts", "anchors"):
        np.testing.assert_array_equal(getattr(graph, field), getattr(reference, field))
    np.testing.assert_array_equal(tree.parents, reference_tree.parents)
    return graph, tree


def _threshold_depth(image, kind):
    components = set()
    height, width = image.shape
    for level in np.unique(image):
        active = (
            image >= level if kind == ComponentTreeKind.MAX_TREE else image <= level
        )
        remaining = set(zip(*np.nonzero(active)))
        while remaining:
            start = remaining.pop()
            component = {start}
            pending = [start]
            while pending:
                row, column = pending.pop()
                for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        pending.append(neighbor)
            components.add(frozenset(row * width + column for row, column in component))

    # Distinct threshold components count strict depth, without node parent links.
    depths = np.full(height * width, -1, dtype=np.int64)
    for component in components:
        for pixel in component:
            depths[pixel] += 1
    return depths.reshape(image.shape)


def _inspect_targets(clean_graph, observed_graph, targets):
    zone, parent_zone, increment = targets.zone, targets.parent_zone, targets.increment
    node_count = observed_graph.levels.size
    assert zone.shape == parent_zone.shape == increment.shape == (node_count,)
    assert set(increment) <= {0, 1}
    represented = set(zone.tolist())
    assert set(parent_zone.tolist()) <= represented

    # Every node of a zone carries the same parent zone and increment.
    zone_parent = {}
    zone_increment = {}
    for node in range(node_count):
        assert (
            zone_parent.setdefault(zone[node], parent_zone[node]) == parent_zone[node]
        )
        assert zone_increment.setdefault(zone[node], increment[node]) == increment[node]

    # One zone is its own parent zone and every zone reaches it without a cycle.
    roots = [z for z in represented if zone_parent[z] == z]
    assert len(roots) == 1
    assert zone_increment[roots[0]] == 0
    zone_depth = {}
    for start in represented:
        current = start
        visited = set()
        depth = 0
        while zone_parent[current] != current:
            assert current not in visited
            visited.add(current)
            depth += zone_increment[current]
            current = zone_parent[current]
        assert current == roots[0]
        zone_depth[start] = depth

    # Equal levels never add depth, whatever the link.
    levels = clean_graph.levels
    for z in represented:
        if levels[z] == levels[zone_parent[z]]:
            assert zone_increment[z] == 0

    node_depth = np.array([zone_depth[z] for z in zone.tolist()], dtype=np.int64)
    return zone_parent, zone_increment, node_depth[observed_graph.label_map]


def _contraction_groups(observed_graph, zone):
    groups = list(range(zone.size))

    def find(node):
        while groups[node] != node:
            node = groups[node]
        return node

    for left, right in observed_graph.edge_index:
        if zone[left] == zone[right]:
            groups[find(int(left))] = find(int(right))
    return np.array([find(node) for node in range(zone.size)])


def _assert_exact_recovery(clean, clean_graph, clean_tree, observed_graph, targets):
    zone_parent, zone_increment, pixel_depth = _inspect_targets(
        clean_graph, observed_graph, targets
    )
    zone = targets.zone

    # Every clean zone survives, and the zone links are the clean node parents.
    assert set(zone_parent) == set(range(clean_graph.levels.size))
    for clean_zone, clean_parent in enumerate(clean_tree.parents):
        assert zone_parent[clean_zone] == clean_parent
        increment = clean_graph.levels[clean_zone] != clean_graph.levels[clean_parent]
        assert zone_increment[clean_zone] == increment

    # Contracting edges inside a zone recovers each clean zone as one group.
    groups = _contraction_groups(observed_graph, zone)
    for clean_zone in range(clean_graph.levels.size):
        assert np.unique(groups[zone == clean_zone]).size == 1
    np.testing.assert_array_equal(zone[observed_graph.label_map], clean_graph.label_map)
    np.testing.assert_array_equal(pixel_depth, _threshold_depth(clean, clean_tree.kind))


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_exhaustive_binary_clean_ternary_observed_recovery(kind):
    clean_cases = []
    for values in product((10, 20), repeat=4):
        image = np.array(values, dtype=np.uint8).reshape(2, 2)
        graph, tree = _checked_graph(image, kind)
        clean_cases.append((image, graph, tree))
    observed_cases = []
    for values in product((0, 10, 20), repeat=4):
        image = np.array(values, dtype=np.uint8).reshape(2, 2)
        graph, _ = _checked_graph(image, kind)
        observed_cases.append(graph)

    refining_count = 0
    mixed_count = 0
    for clean, clean_graph, clean_tree in clean_cases:
        for observed_graph in observed_cases:
            targets, diagnostics = derive_correction_targets(
                clean_graph, clean_tree, observed_graph
            )
            pure = all(
                len(set(clean_graph.label_map[observed_graph.label_map == zone])) == 1
                for zone in range(observed_graph.levels.size)
            )
            if pure:
                _assert_exact_recovery(
                    clean, clean_graph, clean_tree, observed_graph, targets
                )
                assert not diagnostics.missing_clean_zones.any()
                assert not diagnostics.clean_root_missing
                assert diagnostics.artificial_root_attachment_count == 0
                refining_count += 1
            else:
                _inspect_targets(clean_graph, observed_graph, targets)
                mixed_count += 1

    assert refining_count == 630
    assert mixed_count == 666


def _permute_graph(graph, permutation):
    original_ids = np.argsort(permutation)
    return FlatZoneGraph(
        permutation[graph.label_map],
        permutation[graph.edge_index],
        graph.levels[original_ids],
        graph.pixel_counts[original_ids],
        graph.anchors[original_ids],
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_random_refining_recovery_with_permuted_ids(kind):
    rng = np.random.default_rng(20261006)
    for _ in range(32):
        clean = rng.choice(np.array([10, 40, 80], dtype=np.uint8), size=(6, 7))
        # Disjoint intensity ranges prevent observed zones from crossing clean roles.
        observed = clean + rng.integers(0, 6, size=clean.shape, dtype=np.uint8)
        clean_graph, clean_tree = _checked_graph(clean, kind)
        observed_graph, _ = _checked_graph(observed, kind)
        clean_permutation = rng.permutation(clean_graph.levels.size)
        observed_permutation = rng.permutation(observed_graph.levels.size)
        clean_graph = _permute_graph(clean_graph, clean_permutation)
        observed_graph = _permute_graph(observed_graph, observed_permutation)
        clean_tree = ComponentTree(
            kind,
            clean_permutation[clean_tree.parents[np.argsort(clean_permutation)]],
        )
        targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)
        _assert_exact_recovery(clean, clean_graph, clean_tree, observed_graph, targets)


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "clean, observed, zone, parent_zone, increment, depth",
    [
        (
            [[10, 10], [10, 20]],
            [[0, 0], [20, 10]],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 1],
            [[0, 0], [0, 1]],
        ),
        (
            [[10, 10, 10], [10, 20, 30]],
            [[0, 0, 0], [10, 20, 10]],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [[0, 0, 0], [0, 1, 2]],
        ),
        (
            [[10, 20, 30]],
            [[11, 21, 31]],
            [0, 1, 2],
            [0, 0, 1],
            [0, 1, 1],
            [[0, 1, 2]],
        ),
        ([[10]], [[200]], [0], [0], [0], [[0]]),
        (
            [[10, 10, 20, 10, 10]],
            [[11, 15, 15, 15, 12]],
            [0, 0, 2],
            [0, 0, 0],
            [0, 0, 0],
            [[0, 0, 0, 0, 0]],
        ),
        (
            [[10, 10, 10], [20, 20, 20]],
            [[11, 15, 12], [21, 15, 22]],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [[0, 0, 0], [1, 0, 1]],
        ),
    ],
    ids=[
        "quotient-cycle",
        "several-displaced-children",
        "uniform-shift",
        "changed-single-pixel",
        "mixed-bridge",
        "disconnected-output-group",
    ],
)
def test_named_two_dimensional_recovery(
    kind, clean, observed, zone, parent_zone, increment, depth
):
    clean = np.array(clean, dtype=np.uint8)
    observed = np.array(observed, dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean = 255 - clean
        observed = 255 - observed
    clean_graph, clean_tree = _checked_graph(clean, kind)
    observed_graph, _ = _checked_graph(observed, kind)
    targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)
    np.testing.assert_array_equal(targets.zone, zone)
    np.testing.assert_array_equal(targets.parent_zone, parent_zone)
    np.testing.assert_array_equal(targets.increment, increment)
    _, _, pixel_depth = _inspect_targets(clean_graph, observed_graph, targets)
    np.testing.assert_array_equal(pixel_depth, depth)
