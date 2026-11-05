import numpy as np
import pytest
from scipy import ndimage

from letree.component_tree import ComponentTreeKind
from letree.correction.decoder import (
    CorrectionDecodeError,
    DecodeFailureReason,
    contraction_groups,
    decode_groups,
    group_increments,
    hard_vote,
    soft_vote,
    tree_pixel_depth,
)
from letree.generation import (
    AdditiveGaussianNoise,
    GenerationStage,
    SaltAndPepperNoise,
    SquareImageConfig,
    build_generation_sample,
    derive_rng,
    generate_square_image,
)

# The worked example of the zone plan. A one-row image [0 1 5 4 0 0] gives
# observed nodes o0 to o4 on a path, o4 covering the last two pixels
EDGES = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
LABEL_MAP = np.array([[0, 1, 2, 3, 4, 4]])
ZONE = np.array([0, 0, 2, 2, 4])
INCREMENT = np.array([0, 0, 1, 1, 0])
ORACLE_CONTRACTED = np.array([True, False, True, False])
# Every parent zone is A, whose smallest node is o0
ORACLE_POINTERS = np.zeros(5, dtype=np.int64)

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)


def _decode_hard(contracted, pointers, increment, edges=EDGES, label_map=LABEL_MAP):
    groups = contraction_groups(edges, contracted, increment.size)
    return decode_groups(
        groups,
        hard_vote(pointers, groups),
        group_increments(increment.astype(float), groups),
        label_map,
    )


@pytest.mark.parametrize(
    ("contracted", "expected"),
    [
        (ORACLE_CONTRACTED, [0, 0, 1, 1, 2]),
        ([False] * 4, [0, 1, 2, 3, 4]),
        ([True] * 4, [0, 0, 0, 0, 0]),
    ],
    ids=["oracle", "none", "all"],
)
def test_groups_of_the_worked_example(contracted, expected):
    groups = contraction_groups(EDGES, np.array(contracted), 5)
    assert groups.dtype == np.int64
    assert groups.tolist() == expected


def test_groups_are_numbered_by_their_smallest_node_and_isolated_nodes_stay_alone():
    # Edges (4, 1) and (3, 4) join nodes 1, 3 and 4. Node 5 has no edge
    edges = np.array([[4, 3, 0], [1, 4, 2]])
    groups = contraction_groups(edges, np.array([True, True, False]), 6)
    assert groups.tolist() == [0, 1, 2, 1, 1, 3]


def test_hard_vote_counts_argmax_pointers_per_group_pair():
    groups = np.array([0, 0, 1, 1, 2])
    # o3 points into its own group and o4 at o2
    votes = hard_vote(np.array([0, 1, 0, 2, 2]), groups)
    assert votes.tolist() == [[2, 0, 0], [1, 1, 0], [0, 1, 0]]


def test_soft_vote_sums_probability_mass_per_group_pair():
    groups = np.array([0, 0, 1])
    probabilities = np.array(
        [
            [0.5, 0.25, 0.25],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
        ]
    )

    votes = soft_vote(probabilities, groups)

    # Group 0 sends 0.5 + 0.25 + 0.1 + 0.1 to itself and 0.25 + 0.8 to group 1
    np.testing.assert_allclose(votes, [[0.95, 1.05], [0.9, 0.1]])


def test_soft_and_hard_votes_agree_on_one_hot_pointers():
    rng = np.random.default_rng(0)
    groups = np.array([0, 1, 0, 2, 2, 1, 3])
    pointers = rng.integers(0, 7, size=7)
    one_hot = np.eye(7)[pointers]

    np.testing.assert_array_equal(
        soft_vote(one_hot, groups), hard_vote(pointers, groups)
    )


def test_group_increment_is_a_strict_majority_of_the_mean():
    groups = np.array([0, 0, 1, 1, 1, 2])
    values = np.array([1.0, 0.0, 0.9, 0.4, 0.3, 0.51])

    # Group 0 ties at one half and gets 0. Group 1 has mean 1.6 / 3
    assert group_increments(values, groups).tolist() == [0, 1, 1]


def test_oracle_of_the_worked_example_decodes_the_clean_depth():
    decoded = _decode_hard(ORACLE_CONTRACTED, ORACLE_POINTERS, INCREMENT)

    assert decoded.groups.tolist() == [0, 0, 1, 1, 2]
    assert decoded.parents.tolist() == [0, 0, 0]
    assert decoded.increments.tolist() == [0, 1, 0]
    assert decoded.depths.tolist() == [0, 1, 0]
    assert decoded.pixel_depth.tolist() == [[0, 0, 1, 1, 0, 0]]


def test_false_contraction_carries_the_majority_increment_into_the_merged_zone():
    contracted = np.array([True, False, True, True])

    decoded = _decode_hard(contracted, ORACLE_POINTERS, INCREMENT)

    # Increments 1, 1 and 0 have mean 2 / 3, so pixels 4 and 5 move to depth 1
    assert decoded.groups.tolist() == [0, 0, 1, 1, 1]
    assert decoded.pixel_depth.tolist() == [[0, 0, 1, 1, 1, 1]]


def test_two_self_parent_groups_fail_with_root_count():
    # Group 1 votes for itself as well as group 0
    pointers = np.array([0, 0, 2, 2, 0])
    with pytest.raises(CorrectionDecodeError) as error:
        _decode_hard(ORACLE_CONTRACTED, pointers, INCREMENT)
    assert error.value.reason == DecodeFailureReason.ROOT_COUNT


def test_groups_pointing_at_each_other_fail_with_a_cycle():
    # Group 0 is the root, groups 1 and 2 point at each other
    pointers = np.array([0, 0, 4, 4, 2])
    with pytest.raises(CorrectionDecodeError) as error:
        _decode_hard(ORACLE_CONTRACTED, pointers, INCREMENT)
    assert error.value.reason == DecodeFailureReason.NONTRIVIAL_CYCLE


def _threshold_depth(image, kind):
    # Strict depth counted from threshold sets, without any tree. A pixel's depth
    # is the number of distinct connected threshold components containing it,
    # minus one for the root
    components = set()
    for level in np.unique(image):
        active = (
            image >= level if kind == ComponentTreeKind.MAX_TREE else image <= level
        )
        labels, count = ndimage.label(active)
        for label in range(1, count + 1):
            components.add(frozenset(np.flatnonzero(labels == label).tolist()))
    depth = np.full(image.size, -1, dtype=np.int64)
    for component in components:
        depth[list(component)] += 1
    return depth.reshape(image.shape)


def _sample(seed, kind, noise):
    image = generate_square_image(
        SMALL_IMAGES, derive_rng(seed, 0, GenerationStage.SQUARE_IMAGE)
    )
    if noise is None:
        return build_generation_sample(image, kind)
    return build_generation_sample(image, kind, noise, np.random.default_rng(seed))


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_tree_pixel_depth_matches_threshold_components(kind):
    for seed in range(5):
        clean = _sample(seed, kind, None).clean
        depth = tree_pixel_depth(
            clean.component_tree.parents, clean.graph.levels, clean.graph.label_map
        )
        np.testing.assert_array_equal(depth, _threshold_depth(clean.image, kind))


def _oracle(sample):
    # Contractions from the zones, and each node points at the smallest node of
    # its parent zone
    targets, graph = sample.correction_targets, sample.observed.graph
    zone, parent_zone = targets.zone, targets.parent_zone
    edges = graph.edge_index.T
    contracted = zone[edges[0]] == zone[edges[1]]
    smallest_node = {}
    for node, node_zone in enumerate(zone.tolist()):
        smallest_node.setdefault(node_zone, node)
    pointers = np.array([smallest_node[z] for z in parent_zone.tolist()])
    return edges, contracted, pointers, targets.increment


def _is_refining(sample):
    # Every observed zone lies inside one clean zone
    clean = sample.clean.graph.label_map.ravel()
    observed = sample.observed.graph.label_map.ravel()
    pairs = np.unique(np.stack((observed, clean)), axis=1)
    return pairs.shape[1] == observed.max() + 1


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "noise",
    [None, SaltAndPepperNoise(0.05), AdditiveGaussianNoise(standard_deviation=10.0)],
    ids=["clean", "salt_and_pepper", "gaussian"],
)
def test_oracle_decodes_the_clean_depth_on_refining_observations(kind, noise):
    checked = 0
    for seed in range(20):
        sample = _sample(seed, kind, noise)
        if not _is_refining(sample):
            continue
        edges, contracted, pointers, increment = _oracle(sample)

        decoded = _decode_hard(
            contracted, pointers, increment, edges, sample.observed.graph.label_map
        )

        clean = sample.clean
        expected = tree_pixel_depth(
            clean.component_tree.parents, clean.graph.levels, clean.graph.label_map
        )
        np.testing.assert_array_equal(decoded.pixel_depth, expected)
        checked += 1
    assert checked >= 5


def test_oracle_of_a_clean_observation_keeps_every_node_and_its_tree():
    sample = _sample(3, ComponentTreeKind.MAX_TREE, None)
    edges, contracted, pointers, increment = _oracle(sample)

    decoded = _decode_hard(
        contracted, pointers, increment, edges, sample.observed.graph.label_map
    )

    assert not contracted.any()
    np.testing.assert_array_equal(decoded.groups, np.arange(increment.size))
    np.testing.assert_array_equal(decoded.parents, sample.clean.component_tree.parents)


def test_relabelling_observed_nodes_keeps_the_pixel_depth():
    sample = _sample(7, ComponentTreeKind.MAX_TREE, SaltAndPepperNoise(0.05))
    edges, contracted, pointers, increment = _oracle(sample)
    label_map = sample.observed.graph.label_map
    original = _decode_hard(contracted, pointers, increment, edges, label_map)

    # New id of each old node, applied to every array that names nodes
    new_id = np.random.default_rng(1).permutation(increment.size)
    renamed_pointers = np.empty_like(pointers)
    renamed_pointers[new_id] = new_id[pointers]
    renamed_increment = np.empty_like(increment)
    renamed_increment[new_id] = increment
    renamed = _decode_hard(
        contracted,
        renamed_pointers,
        renamed_increment,
        new_id[edges],
        new_id[label_map],
    )

    np.testing.assert_array_equal(renamed.pixel_depth, original.pixel_depth)
