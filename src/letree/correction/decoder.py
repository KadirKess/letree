"""Hard decoding of contracted groups, their parents and depth increments."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


class DecodeFailureReason(StrEnum):
    ROOT_COUNT = "root_count"
    NONTRIVIAL_CYCLE = "nontrivial_cycle"


class CorrectionDecodeError(ValueError):
    def __init__(self, reason: DecodeFailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class DecodedCorrection:
    # Output nodes are the groups of contracted observed nodes
    groups: NDArray[np.int64]
    parents: NDArray[np.int64]
    increments: NDArray[np.int64]
    depths: NDArray[np.int64]
    pixel_depth: NDArray[np.int64]


def contraction_groups(
    rag_edge_index: NDArray[np.integer],
    contracted: NDArray[np.bool_],
    node_count: int,
) -> NDArray[np.int64]:
    source, destination = rag_edge_index[:, contracted]
    # Sparse, because a dense adjacency takes about 115 MB at 3,800 nodes
    adjacency = coo_matrix(
        (np.ones(source.size), (source, destination)), shape=(node_count, node_count)
    )
    _, groups = connected_components(adjacency, directed=False)
    return groups.astype(np.int64)


def hard_vote(
    pointers: NDArray[np.integer], groups: NDArray[np.int64]
) -> NDArray[np.int64]:
    group_count = groups.max() + 1
    votes = np.zeros((group_count, group_count), dtype=np.int64)
    np.add.at(votes, (groups, groups[pointers]), 1)
    return votes


def soft_vote(
    pointer_probabilities: NDArray[np.floating], groups: NDArray[np.int64]
) -> NDArray[np.float64]:
    node_count = groups.size
    # With membership M[i, g], the votes are M transposed times P times M
    membership = csr_matrix(
        (np.ones(node_count), (np.arange(node_count), groups)),
        shape=(node_count, groups.max() + 1),
    )
    return np.asarray((membership.T @ pointer_probabilities) @ membership)


def group_increments(
    values: NDArray[np.floating], groups: NDArray[np.int64]
) -> NDArray[np.int64]:
    # Strictly above one half, so a tie gives 0
    means = np.bincount(groups, weights=values) / np.bincount(groups)
    return (means > 0.5).astype(np.int64)


def decode_groups(
    groups: NDArray[np.int64],
    votes: NDArray[np.number],
    increments: NDArray[np.int64],
    observed_label_map: NDArray[np.integer],
) -> DecodedCorrection:
    # A tie goes to the smallest group id
    parents = votes.argmax(axis=1)
    root = _find_root(parents)
    depths = _accumulate_depths(parents, increments, root)
    pixel_depth = depths[groups[observed_label_map]]
    return DecodedCorrection(groups, parents, increments, depths, pixel_depth)


def tree_pixel_depth(
    node_parents: NDArray[np.integer],
    levels: NDArray[np.integer],
    label_map: NDArray[np.integer],
) -> NDArray[np.int64]:
    # A level change adds one depth step and an equal-level link adds none
    increments = (levels != levels[node_parents]).astype(np.int64)
    root = _find_root(node_parents)
    depths = _accumulate_depths(node_parents, increments, root)
    return depths[label_map]


def _find_root(parents: NDArray[np.integer]) -> int:
    n = parents.size
    roots = np.flatnonzero(parents == np.arange(n))
    if roots.size != 1:
        raise CorrectionDecodeError(
            DecodeFailureReason.ROOT_COUNT,
            message=f"Expected exactly one self-parent root, found {roots.size}.",
        )
    root = int(roots[0])

    # State 0 is unvisited, 1 is on the current chain and 2 is known to reach the
    # root. Reaching a 1 again means the chain looped back on itself.
    state = np.zeros(n, dtype=np.int8)
    state[root] = 2
    for start in range(n):
        if state[start] == 2:
            continue

        path = []
        node = start
        while state[node] == 0:
            path.append(node)
            state[node] = 1
            node = parents[node]

        if state[node] == 1:
            raise CorrectionDecodeError(
                DecodeFailureReason.NONTRIVIAL_CYCLE,
                message=f"Parent chain from node {start} loops back to node {node}.",
            )

        state[path] = 2

    return root


def _accumulate_depths(
    parents: NDArray[np.int64],
    increments: NDArray[np.int64],
    root: int,
) -> NDArray[np.int64]:
    output_node_count = parents.size
    children = [[] for _ in range(output_node_count)]

    # The root's self-link is not a child edge
    for node in range(output_node_count):
        parent = parents[node]
        if parent != node:
            children[parent].append(node)

    # Depth comes from increments, so a child with increment 0 sits at its
    # parent's depth. A parent's depth is set before its children are pushed.
    depths = np.zeros(output_node_count, dtype=np.int64)
    pending = [root]
    while pending:
        node = pending.pop()
        for child in children[node]:
            depths[child] = depths[node] + increments[child]
            pending.append(child)

    return depths
