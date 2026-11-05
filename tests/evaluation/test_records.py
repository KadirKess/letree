import numpy as np
import pytest

from letree.component_tree import ComponentTreeKind
from letree.data.families import build_family
from letree.evaluation.records import (
    GraphPrediction,
    clean_reference,
    decode_source,
    oracle_pointers,
    score_record,
)
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


def _records(source, observation, clean, prediction=None):
    counts, hierarchy = decode_source(source, observation, prediction)
    return counts | score_record(hierarchy, observation, clean_reference(clean))


def _target_prediction(observation, soft_pointers=False):
    """A model output that puts all its weight on the targets."""
    targets = observation.targets
    node_count = targets.zone.size
    source, destination = observation.tree_image.graph.edge_index.T
    contracted = targets.zone[source] == targets.zone[destination]
    pointers = oracle_pointers(targets.zone, targets.parent_zone)
    if soft_pointers:
        # Spread evenly over the parent zone, which keeps the vote on it
        in_parent = targets.zone[None, :] == targets.parent_zone[:, None]
        probabilities = in_parent / in_parent.sum(axis=1, keepdims=True)
    else:
        probabilities = np.eye(node_count)[pointers]
    return GraphPrediction(
        contraction_logits=np.where(contracted, 3.0, -3.0),
        increment_probabilities=np.where(targets.increment == 1, 0.9, 0.1),
        pointer_probabilities=probabilities,
        pointers=pointers,
        ties=0,
    )


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


@pytest.mark.parametrize("soft_pointers", [False, True])
def test_model_votes_on_the_targets_match_the_oracle(soft_pointers):
    for family in _families(10):
        for observation in family:
            prediction = _target_prediction(observation, soft_pointers)
            oracle = _records("oracle", observation, family[0])
            for source in ("model_soft", "model_hard"):
                record = _records(source, observation, family[0], prediction)
                assert record == oracle, source
            assert oracle["root_claims"] == 1


def test_self_pointers_give_an_invalid_record():
    family = _families(1)[0]
    observation = family[2]
    node_count = observation.targets.zone.size
    # No edge contracts and every node points to itself, so every group is a root
    prediction = GraphPrediction(
        contraction_logits=np.full(len(observation.tree_image.graph.edge_index), -1.0),
        increment_probabilities=np.zeros(node_count),
        pointer_probabilities=np.eye(node_count),
        pointers=np.arange(node_count),
        ties=0,
    )

    for source in ("model_soft", "model_hard"):
        record = _records(source, observation, family[0], prediction)
        assert record["valid"] is False
        assert record["failure"] == "root_count"
        assert record["exact"] is False
        assert record["groups"] == record["root_claims"] == node_count
        assert "lost_zones" not in record


def test_a_logit_of_zero_does_not_contract():
    family = _families(1)[0]
    observation = family[0]
    prediction = _target_prediction(observation)
    prediction.contraction_logits[:] = 0.0
    record = _records("model_hard", observation, family[0], prediction)
    assert record["groups"] == observation.targets.zone.size
