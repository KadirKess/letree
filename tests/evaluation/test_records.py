import numpy as np
import pytest

from letree.component_tree import ComponentTreeKind
from letree.data.families import build_family
from letree.evaluation.records import clean_reference, score_record
from letree.evaluation.scores import tree_correction
from letree.generation import AdditiveGaussianNoise, SaltAndPepperNoise

NOISES = (SaltAndPepperNoise(probability=0.1), AdditiveGaussianNoise(10.0))

# Background 10, a square of 20 and a pixel of 30 inside it
NESTED = np.array(
    [
        [10, 10, 10, 10],
        [10, 20, 20, 10],
        [10, 20, 30, 10],
        [10, 10, 10, 10],
    ],
    dtype=np.uint8,
)


def _families(count, kind=ComponentTreeKind.MAX_TREE):
    rng = np.random.default_rng(20261103)
    images = [
        rng.choice(np.array([10, 60, 120], np.uint8), (6, 6)) for _ in range(count)
    ]
    return [
        build_family(image, 7, i, kind, NOISES, 4) for i, image in enumerate(images)
    ]


def _records(source, observation, clean):
    tree = observation.tree_image
    if source == "oracle":
        prediction = (
            observation.targets.corrective_parents,
            observation.targets.actions,
        )
    else:
        prediction = tree_correction(tree.component_tree.parents, tree.graph.levels)
    return score_record(*prediction, observation, clean_reference(clean))


@pytest.mark.parametrize(
    "kind, depth",
    [
        (
            ComponentTreeKind.MAX_TREE,
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
        ),
        (
            ComponentTreeKind.MIN_TREE,
            [[2, 2, 2, 2], [2, 1, 1, 2], [2, 1, 0, 2], [2, 2, 2, 2]],
        ),
    ],
)
def test_clean_reference_of_nested_squares(kind, depth):
    clean = build_family(NESTED, 0, 0, kind, NOISES, 0)[0]
    labels, pixel_depth, root = clean_reference(clean)
    np.testing.assert_array_equal(pixel_depth, depth)
    # The root is the zone at depth zero
    assert labels[pixel_depth == 0][0] == root
    parents = clean.tree_image.component_tree.parents
    assert parents[root] == root


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_oracle_recovers_every_refining_sample(kind):
    refining = 0
    for family in _families(20, kind):
        for observation in family:
            record = _records("oracle", observation, family[0])
            assert record["valid"]
            if record["refining"]:
                refining += 1
                assert record["exact"]
    # The clean samples are refining, and salt-and-pepper adds some more
    assert refining > 20


def test_observed_tree_recovers_every_clean_sample():
    for family in _families(10):
        record = _records("observed", family[0], family[0])
        assert record["refining"] and record["exact"]
        assert record["all/depth_error"] == 0


def test_pure_and_mixed_pixels_cover_the_image():
    for family in _families(10):
        for observation in family:
            record = _records("observed", observation, family[0])
            assert record["pure/pixels"] + record["mixed/pixels"] == 36
            assert record["all/pixels"] == 36
            assert record["refining"] == (record["mixed/pixels"] == 0)
            for field in ("misassigned", "depth_error", "exact_depth"):
                assert (
                    record[f"pure/{field}"] + record[f"mixed/{field}"]
                    == (record[f"all/{field}"])
                )


def test_two_roots_give_an_invalid_record():
    family = _families(1)[0]
    observation = family[2]
    parents = observation.tree_image.component_tree.parents.copy()
    actions = observation.targets.actions.copy()
    child = int(np.flatnonzero(parents != np.arange(parents.size))[0])
    parents[child] = child
    actions[child] = 1

    record = score_record(parents, actions, observation, clean_reference(family[0]))

    assert record["valid"] is False
    assert record["failure"] == "root_count"
    assert record["exact"] is False
    assert "lost_zones" not in record
