"""Bulk conversion from Higra components to node parents for production."""

import higra as hg
import numpy as np
from numpy.typing import NDArray

from letree.component_tree.higra.rag import HigraRagContext
from letree.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph


def _choose_component_representatives(
    graph: FlatZoneGraph,
    rag_nodes: NDArray[np.int64],
    component_of_leaf: NDArray[np.int64],
    component_count: int,
) -> NDArray[np.int64]:
    """Choose one RAG node for every Higra component in bulk."""

    candidate_counts = np.asarray(graph.pixel_counts[rag_nodes], dtype=np.int64)

    # Each component slot receives the greatest count among its introduced leaves
    largest_counts = np.zeros(component_count, dtype=np.int64)
    np.maximum.at(largest_counts, component_of_leaf, candidate_counts)

    # Only leaves tied for the greatest count remain eligible representatives
    has_largest_count = candidate_counts == largest_counts[component_of_leaf]
    eligible_components = component_of_leaf[has_largest_count]
    eligible_anchors = np.asarray(
        graph.anchors[rag_nodes[has_largest_count]],
        dtype=np.int64,
    )

    # Resolve each remaining tie with the smallest row-major pixel anchor
    smallest_anchors = np.full(
        component_count,
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )
    np.minimum.at(smallest_anchors, eligible_components, eligible_anchors)

    candidate_anchors = np.asarray(graph.anchors[rag_nodes], dtype=np.int64)
    is_representative = has_largest_count & (
        candidate_anchors == smallest_anchors[component_of_leaf]
    )

    # Unique anchors guarantee exactly one selected leaf for each component
    representatives = np.empty(component_count, dtype=np.int64)
    representatives[component_of_leaf[is_representative]] = rag_nodes[is_representative]
    return representatives


def higra_to_node_parents_vectorized(
    graph: FlatZoneGraph,
    higra_tree: hg.Tree,
    context: HigraRagContext,
    kind: ComponentTreeKind,
) -> ComponentTree:
    """Convert Higra components to node parents with grouped array operations."""

    kind = ComponentTreeKind(kind)
    leaf_count = graph.levels.size
    if context.higra_to_rag_node.shape != (leaf_count,):
        raise ValueError("The Higra leaf map must contain one entry per RAG node.")

    higra_parents = np.asarray(higra_tree.parents(), dtype=np.int64)
    component_count = higra_parents.size - leaf_count
    if component_count <= 0:
        raise ValueError("The Higra tree must contain an internal component node.")

    # Higra leaves come first, so subtracting their count gives dense component IDs
    component_of_leaf = higra_parents[:leaf_count] - leaf_count
    if np.any((component_of_leaf < 0) | (component_of_leaf >= component_count)):
        raise ValueError("Every Higra leaf must point to an internal component node.")

    # A canonical Higra component introduces at least one direct leaf child
    introduced_counts = np.bincount(
        component_of_leaf,
        minlength=component_count,
    )
    if np.any(introduced_counts == 0):
        raise ValueError("Every canonical Higra component must introduce a RAG leaf.")

    rag_nodes = context.higra_to_rag_node
    representatives = _choose_component_representatives(
        graph,
        rag_nodes,
        component_of_leaf,
        component_count,
    )

    # Point every introduced leaf to the representative of its own component
    parents = np.empty(leaf_count, dtype=np.int64)
    parents[rag_nodes] = representatives[component_of_leaf]

    # Replace each representative's self-link with its enclosing representative
    # The Higra root encloses itself, which naturally preserves the root's self-parent
    enclosing_components = higra_parents[leaf_count:] - leaf_count
    parents[representatives] = representatives[enclosing_components]

    return ComponentTree(kind=kind, parents=parents)
