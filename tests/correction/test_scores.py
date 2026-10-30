import numpy as np
import pytest

from letree.component_tree import ComponentTreeKind
from letree.correction.decoder import decode_correction
from letree.correction.scores import (
    CorrectionScores,
    score_correction,
    tree_correction,
)
from letree.generation import derive_correction_targets
from letree.generation.samples import _build_production_tree


def _ints(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _tree_depth(image, kind):
    graph, tree = _build_production_tree(np.asarray(image, dtype=np.uint8), kind)
    decoded = decode_correction(
        *tree_correction(tree.parents, graph.levels), graph.label_map
    )
    return graph, tree, decoded


def _root(parents) -> int:
    return int(np.flatnonzero(parents == np.arange(parents.size))[0])


def _assert_scores(scores, **expected) -> None:
    assert isinstance(scores, CorrectionScores)
    for name, value in expected.items():
        if isinstance(value, bool):
            assert getattr(scores, name) is value, name
        else:
            assert getattr(scores, name) == pytest.approx(value), name


@pytest.mark.parametrize(
    "kind, depth",
    [
        (ComponentTreeKind.MAX_TREE, [[0, 1, 0]]),
        (ComponentTreeKind.MIN_TREE, [[1, 0, 1]]),
    ],
)
def test_tree_correction_decodes_to_tree_depth(kind, depth):
    _, _, decoded = _tree_depth([[10, 20, 10]], kind)
    np.testing.assert_array_equal(decoded.pixel_depth, depth)


def test_tree_correction_keeps_parents_and_marks_level_changes():
    parents, actions = tree_correction(_ints([0, 0, 1, 1]), _ints([10, 20, 20, 30]))
    assert parents.dtype == np.int64 and actions.dtype == np.int64
    np.testing.assert_array_equal(parents, [0, 0, 1, 1])
    # Root and equal-level link stay at depth, level changes add one
    np.testing.assert_array_equal(actions, [1, 2, 1, 2])


# Clean [[10, 10, 10], [20, 20, 20]] and observed [[11, 15, 12], [21, 15, 22]],
# max-tree, as worked in the guide
CLEAN_LABELS = _ints([[0, 0, 0], [1, 1, 1]])
CLEAN_DEPTH = _ints([[0, 0, 0], [1, 1, 1]])


def test_worked_example_oracle():
    scores = score_correction(
        _ints([[0, 0, 0], [1, 0, 1]]),
        _ints([[0, 0, 0], [1, 0, 1]]),
        CLEAN_LABELS,
        CLEAN_DEPTH,
        0,
    )
    _assert_scores(
        scores,
        lost_clean_zone_fraction=0.0,
        lost_clean_area_fraction=0.0,
        misassigned_pixel_fraction=1 / 6,
        depth_mae=1 / 6,
        exact_depth_fraction=5 / 6,
        clean_root_lost=False,
    )


def test_worked_example_observed_tree_baseline():
    scores = score_correction(
        _ints([[0, 1, 2], [3, 1, 4]]),
        _ints([[0, 2, 1], [3, 2, 3]]),
        CLEAN_LABELS,
        CLEAN_DEPTH,
        0,
    )
    _assert_scores(
        scores,
        lost_clean_zone_fraction=0.0,
        lost_clean_area_fraction=0.0,
        misassigned_pixel_fraction=1 / 6,
        depth_mae=8 / 6,
        exact_depth_fraction=1 / 6,
        clean_root_lost=False,
    )


def test_worked_example_through_production_pipeline():
    kind = ComponentTreeKind.MAX_TREE
    clean_graph, clean_tree, clean = _tree_depth([[10, 10, 10], [20, 20, 20]], kind)
    observed_graph, observed_tree, baseline = _tree_depth(
        [[11, 15, 12], [21, 15, 22]], kind
    )
    targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)
    oracle = decode_correction(
        targets.corrective_parents, targets.actions, observed_graph.label_map
    )
    np.testing.assert_array_equal(clean.pixel_depth, CLEAN_DEPTH)
    np.testing.assert_array_equal(oracle.pixel_depth, [[0, 0, 0], [1, 0, 1]])
    np.testing.assert_array_equal(baseline.pixel_depth, [[0, 2, 1], [3, 2, 3]])

    scores = score_correction(
        baseline.observed_to_output[observed_graph.label_map],
        baseline.pixel_depth,
        clean_graph.label_map,
        clean.pixel_depth,
        _root(clean_tree.parents),
    )
    assert scores.depth_mae == pytest.approx(8 / 6)


def test_perfect_decode():
    scores = score_correction(
        CLEAN_LABELS + 7, CLEAN_DEPTH, CLEAN_LABELS, CLEAN_DEPTH, 0
    )
    _assert_scores(
        scores,
        lost_clean_zone_fraction=0.0,
        lost_clean_area_fraction=0.0,
        misassigned_pixel_fraction=0.0,
        depth_mae=0.0,
        exact_depth_fraction=1.0,
        clean_root_lost=False,
    )


def test_merging_everything_loses_the_smaller_zone():
    scores = score_correction(
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((2, 3), dtype=np.int64),
        CLEAN_LABELS,
        CLEAN_DEPTH,
        0,
    )
    # Both zones have three pixels, so the tie keeps zone 0 and loses zone 1
    _assert_scores(
        scores,
        lost_clean_zone_fraction=1 / 2,
        lost_clean_area_fraction=3 / 6,
        misassigned_pixel_fraction=3 / 6,
        depth_mae=3 / 6,
        exact_depth_fraction=3 / 6,
        clean_root_lost=False,
    )


def test_merging_everything_can_lose_the_root():
    scores = score_correction(
        _ints([[0, 0, 0]]),
        _ints([[0, 0, 0]]),
        _ints([[0, 1, 1]]),
        _ints([[0, 1, 1]]),
        0,
    )
    _assert_scores(
        scores,
        lost_clean_zone_fraction=1 / 2,
        lost_clean_area_fraction=1 / 3,
        misassigned_pixel_fraction=1 / 3,
        depth_mae=2 / 3,
        exact_depth_fraction=1 / 3,
        clean_root_lost=True,
    )


def test_even_split_goes_to_the_smaller_clean_zone():
    scores = score_correction(
        _ints([[5, 5]]), _ints([[0, 0]]), _ints([[0, 1]]), _ints([[0, 1]]), 0
    )
    _assert_scores(
        scores,
        lost_clean_zone_fraction=1 / 2,
        lost_clean_area_fraction=1 / 2,
        misassigned_pixel_fraction=1 / 2,
        clean_root_lost=False,
    )


def test_group_numbering_does_not_change_scores():
    rng = np.random.default_rng(20261026)
    clean_labels = rng.integers(0, 4, size=(5, 6))
    clean_labels.flat[:4] = np.arange(4)
    clean_depth = rng.integers(0, 3, size=(5, 6))
    groups = rng.integers(0, 6, size=(5, 6))
    depth = rng.integers(0, 3, size=(5, 6))
    renumbered = rng.permutation(np.arange(100, 106))[groups]
    assert score_correction(groups, depth, clean_labels, clean_depth, 0) == (
        score_correction(renumbered, depth, clean_labels, clean_depth, 0)
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_oracle_scores_agree_with_stored_diagnostics(kind):
    rng = np.random.default_rng(20261026)
    root_lost_count = 0
    for _ in range(100):
        clean_graph, clean_tree, clean = _tree_depth(
            rng.choice(np.array([10, 40, 80], dtype=np.uint8), size=(5, 6)), kind
        )
        observed_graph, _ = _build_production_tree(
            rng.choice(np.array([0, 10, 40, 80], dtype=np.uint8), size=(5, 6)), kind
        )
        targets, diagnostics = derive_correction_targets(
            clean_graph, clean_tree, observed_graph
        )
        oracle = decode_correction(
            targets.corrective_parents, targets.actions, observed_graph.label_map
        )
        scores = score_correction(
            oracle.observed_to_output[observed_graph.label_map],
            oracle.pixel_depth,
            clean_graph.label_map,
            clean.pixel_depth,
            _root(clean_tree.parents),
        )

        lost = diagnostics.missing_clean_zones
        area = observed_graph.label_map.size
        assert scores.misassigned_pixel_fraction == pytest.approx(
            np.sum(observed_graph.pixel_counts * (1 - diagnostics.overlap_purity))
            / area
        )
        assert scores.lost_clean_zone_fraction == pytest.approx(lost.mean())
        assert scores.lost_clean_area_fraction == pytest.approx(
            clean_graph.pixel_counts[lost].sum() / area
        )
        assert scores.clean_root_lost == bool(diagnostics.clean_root_missing)
        root_lost_count += diagnostics.clean_root_missing

    # The random images must include lost roots, or the flag check means little
    assert root_lost_count > 0
