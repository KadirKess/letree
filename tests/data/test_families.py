import json
from dataclasses import replace

import numpy as np
import pytest

from letree.component_tree import ComponentTreeKind
from letree.data.families import (
    Observation,
    build_family,
    check_family,
    family_path,
    generate_dataset,
    load_family,
    load_observation,
    save_family,
)
from letree.data.sources import read_image, source_images, write_square_source
from letree.generation import (
    NOISE_KIND_NAMES,
    AdditiveGaussianNoise,
    CorrectionTargets,
    GenerationStage,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    SquareImageConfig,
    build_generation_sample,
    corrupt_image,
    derive_correction_targets,
    derive_rng,
    generate_square_image,
)

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)
NOISE = SaltAndPepperNoise(probability=0.05)
GAUSSIAN = AdditiveGaussianNoise(standard_deviation=10.0)
ALL_NOISES = (NOISE, GAUSSIAN, SpeckleNoise(), PoissonNoise())


def _family(
    sample_index=5,
    base_seed=3,
    noises=(NOISE,),
    noisy_count=2,
    kind=ComponentTreeKind.MAX_TREE,
):
    rng = derive_rng(base_seed, sample_index, GenerationStage.SQUARE_IMAGE)
    image = generate_square_image(SMALL_IMAGES, rng)
    return build_family(image, base_seed, sample_index, kind, noises, noisy_count)


def _square_dataset(directory, counts, noises=(NOISE,), noisy_counts=None, **options):
    write_square_source(directory / "source", 3, counts, SMALL_IMAGES)
    generate_dataset(
        directory / "data",
        directory / "source",
        3,
        options.get("kind", ComponentTreeKind.MAX_TREE),
        noises,
        noisy_counts or {split: 2 for split in counts},
        options.get("workers", 1),
    )
    return directory / "data"


def _assert_observations_equal(actual, expected):
    assert (actual.noise, actual.realization) == (expected.noise, expected.realization)
    a, e = actual.tree_image, expected.tree_image
    assert a.component_tree.kind == e.component_tree.kind
    pairs = [
        (a.image, e.image),
        (a.component_tree.parents, e.component_tree.parents),
        *[(getattr(a.graph, n), getattr(e.graph, n)) for n in vars(e.graph)],
        *[
            (getattr(actual.targets, n), getattr(expected.targets, n))
            for n in vars(expected.targets)
        ],
    ]
    for actual_array, expected_array in pairs:
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_array_equal(actual_array, expected_array)
    for name, value in vars(expected.diagnostics).items():
        stored = getattr(actual.diagnostics, name)
        assert type(stored) is type(value), name
        np.testing.assert_array_equal(stored, value)


def test_family_is_reproducible_from_its_seeds():
    for a, b in zip(_family(), _family(), strict=True):
        _assert_observations_equal(a, b)


def _kinds(family):
    return [(o.noise, o.realization) for o in family]


def test_family_is_clean_then_kinds_in_rotation_from_the_sample_index():
    # Sample 5 among four kinds starts at kind 5 mod 4, the Gaussian one
    assert _kinds(_family(noises=ALL_NOISES, noisy_count=2)) == [
        ("clean", 0),
        ("additive_gaussian", 0),
        ("speckle", 0),
    ]
    assert _kinds(_family(noises=ALL_NOISES, noisy_count=6)) == [
        ("clean", 0),
        ("additive_gaussian", 0),
        ("speckle", 0),
        ("poisson", 0),
        ("salt_and_pepper", 0),
        ("additive_gaussian", 1),
        ("speckle", 1),
    ]
    assert _kinds(_family(noises=ALL_NOISES, noisy_count=0)) == [("clean", 0)]


def test_a_kind_drawn_twice_gets_a_new_realization():
    family = _family(noises=(NOISE,), noisy_count=2)

    assert _kinds(family) == [
        ("clean", 0),
        ("salt_and_pepper", 0),
        ("salt_and_pepper", 1),
    ]
    assert not np.array_equal(family[1].tree_image.image, family[2].tree_image.image)


@pytest.mark.parametrize("noisy_count", [1, 2, 3])
def test_consecutive_images_share_every_kind_equally(noisy_count):
    # Over four consecutive images each kind is drawn noisy_count times
    counts = {}
    for sample_index in range(10, 14):
        for o in _family(sample_index, noises=ALL_NOISES, noisy_count=noisy_count)[1:]:
            counts[o.noise] = counts.get(o.noise, 0) + 1
    assert counts == {NOISE_KIND_NAMES[type(n)]: noisy_count for n in ALL_NOISES}


@pytest.mark.parametrize(
    ("noise", "stage"),
    [
        (NOISE, GenerationStage.SALT_AND_PEPPER_NOISE),
        (GAUSSIAN, GenerationStage.ADDITIVE_GAUSSIAN_NOISE),
        (SpeckleNoise(), GenerationStage.SPECKLE_NOISE),
        (PoissonNoise(), GenerationStage.POISSON_NOISE),
    ],
)
def test_each_observation_draws_from_its_kind_stream(noise, stage):
    # Rebuilt from the documented streams, not from the family builder, so the
    # single-noise datasets generated before mixing stay reproducible
    family = _family(noises=ALL_NOISES, noisy_count=8)
    clean = generate_square_image(
        SMALL_IMAGES, derive_rng(3, 5, GenerationStage.SQUARE_IMAGE)
    )
    np.testing.assert_array_equal(family[0].tree_image.image, clean)

    observations = [o for o in family if o.noise == NOISE_KIND_NAMES[type(noise)]]
    assert [o.realization for o in observations] == [0, 1]
    for observation in observations:
        rng = derive_rng(3, 5, stage, realization_index=observation.realization)
        np.testing.assert_array_equal(
            observation.tree_image.image, corrupt_image(clean, noise, rng)
        )


def test_adding_a_noise_kind_leaves_other_kinds_unchanged():
    alone = _family(noises=(GAUSSIAN,))
    mixed = _family(noises=(NOISE, GAUSSIAN), noisy_count=4)

    mixed_gaussian = [o for o in mixed if o.noise == "additive_gaussian"]
    for a, b in zip(alone[1:], mixed_gaussian, strict=True):
        _assert_observations_equal(a, b)


def test_two_configurations_of_one_kind_are_rejected():
    # They would read the same stream, so their noise would be correlated
    with pytest.raises(ValueError, match="only once"):
        _family(noises=(GAUSSIAN, AdditiveGaussianNoise(standard_deviation=5.0)))


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_family_round_trip_is_exact(tmp_path, kind):
    family = _family(noises=ALL_NOISES, noisy_count=8, kind=kind)
    path = tmp_path / "family.npz"

    save_family(path, family, "square.png")

    loaded = load_family(path)
    assert len(loaded) == len(family) == 9
    for position, (actual, expected) in enumerate(zip(loaded, family, strict=True)):
        _assert_observations_equal(actual, expected)
        _assert_observations_equal(load_observation(path, position), expected)


def test_loading_and_adaptation_do_not_rederive_targets(tmp_path, monkeypatch):
    import letree.generation.samples as sample_module
    import letree.generation.targets as target_module
    from letree.data.graphs import to_graph

    family = _family()
    path = tmp_path / "family.npz"
    save_family(path, family, "square.png")

    def forbidden(*args, **kwargs):
        pytest.fail("Stored correction supervision must not be rederived.")

    monkeypatch.setattr(sample_module, "derive_correction_targets", forbidden)
    monkeypatch.setattr(target_module, "derive_correction_targets", forbidden)
    for observation in load_family(path):
        graph = to_graph(observation.tree_image, observation.targets)
        np.testing.assert_array_equal(
            graph.corrective_parents.numpy(), observation.targets.corrective_parents
        )


def test_diagnostics_keep_their_clean_and_observed_scopes(tmp_path):
    # The observed image merges the first two clean zones, so one of them has no
    # observed representative
    kind = ComponentTreeKind.MAX_TREE
    clean = build_generation_sample(np.array([[10, 20, 30]], dtype=np.uint8), kind)
    observed = build_generation_sample(np.array([[25, 25, 35]], dtype=np.uint8), kind)
    targets, diagnostics = derive_correction_targets(
        clean.clean.graph, clean.clean.component_tree, observed.observed.graph
    )
    family = [
        Observation(
            "clean",
            0,
            clean.observed,
            clean.correction_targets,
            clean.correction_diagnostics,
        ),
        Observation("additive_gaussian", 0, observed.observed, targets, diagnostics),
    ]
    save_family(tmp_path / "family.npz", family, "square.png")

    loaded = load_family(tmp_path / "family.npz")[1]
    np.testing.assert_array_equal(loaded.targets.corrective_parents, [0, 0])
    np.testing.assert_array_equal(loaded.targets.actions, [1, 2])
    np.testing.assert_array_equal(
        loaded.diagnostics.missing_clean_zones, [False, True, False]
    )
    assert loaded.diagnostics.overlap_purity.shape == (2,)


def test_one_realization_index_may_repeat_across_noise_kinds():
    check_family(_family(noises=(NOISE, GAUSSIAN), noisy_count=2))


def test_a_repeated_realization_within_one_kind_is_rejected(tmp_path):
    family = _family()
    family[2] = replace(family[2], realization=0)

    with pytest.raises(ValueError, match="unique within a noise kind"):
        save_family(tmp_path / "family.npz", family, "square.png")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("parents", "actions", "message"),
    [
        ([0, 0, 3], [1, 0, 0], "local observed node"),
        ([0, 1, 0], [1, 1, 0], "one self-parented"),
        ([0, 0, 0], [0, 0, 0], "KEEP_ZERO root"),
        ([0, 2, 1], [1, 0, 0], "cycle"),
    ],
    ids=["out_of_range", "two_roots", "merging_root", "cycle"],
)
def test_invalid_targets_are_rejected_before_writing(parents, actions, message):
    family = _family()
    family[1] = replace(
        family[1],
        targets=CorrectionTargets(np.array(parents), np.array(actions)),
    )

    with pytest.raises(ValueError, match=message):
        check_family(family)


@pytest.mark.parametrize("workers", [1, 2])
def test_families_follow_their_source_images_and_splits(tmp_path, workers):
    counts = {"train": 3, "val": 2, "test": 1}
    root = _square_dataset(tmp_path, counts, workers=workers)

    names = {
        split: sorted(p.name for p in (root / split).glob("*.npz")) for split in counts
    }
    assert names == {
        "train": [f"sample_{i:08d}.npz" for i in range(3)],
        "val": [f"sample_{i:08d}.npz" for i in range(3, 5)],
        "test": ["sample_00000005.npz"],
    }
    # Going through PNG files gives the same families as building them directly
    path = family_path(root, "val", 3)
    for a, b in zip(load_family(path), _family(3), strict=True):
        _assert_observations_equal(a, b)
    with np.load(path) as arrays:
        assert str(arrays["source"]) == "square_00000003.png"


def test_manifest_records_the_dataset_and_its_source(tmp_path):
    root = _square_dataset(
        tmp_path, {"train": 1}, ALL_NOISES, kind=ComponentTreeKind.MIN_TREE
    )

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["base_seed"] == 3
    assert manifest["component_tree_kind"] == "min_tree"
    assert manifest["correction_target_policy"] == "pixel_count_representative_v1"
    assert manifest["noise_configs"]["additive_gaussian"]["standard_deviation"] == 10.0
    assert list(manifest["noise_configs"]) == [
        "salt_and_pepper",
        "additive_gaussian",
        "speckle",
        "poisson",
    ]
    assert manifest["noisy_counts"] == {"train": 2}
    assert manifest["split_sample_counts"] == {"train": 1}
    assert manifest["source"]["source"] == "square_images"
    assert manifest["source"]["image_config"]["height"] == 16


def test_an_existing_source_or_dataset_is_never_overwritten(tmp_path):
    _square_dataset(tmp_path, {"train": 1})

    with pytest.raises(FileExistsError):
        write_square_source(tmp_path / "source", 3, {"train": 1}, SMALL_IMAGES)
    with pytest.raises(FileExistsError):
        generate_dataset(
            tmp_path / "data",
            tmp_path / "source",
            3,
            ComponentTreeKind.MAX_TREE,
            (NOISE,),
            {"train": 1},
        )


def test_each_split_gets_its_own_noisy_count(tmp_path):
    root = _square_dataset(
        tmp_path,
        {"train": 2, "val": 1},
        ALL_NOISES,
        noisy_counts={"train": 1, "val": 4},
    )

    assert _kinds(load_family(family_path(root, "train", 1))) == [
        ("clean", 0),
        ("additive_gaussian", 0),
    ]
    val = load_family(family_path(root, "val", 2))
    assert sorted(o.noise for o in val[1:]) == sorted(
        NOISE_KIND_NAMES[type(n)] for n in ALL_NOISES
    )


def test_a_split_without_a_noisy_count_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="val"):
        _square_dataset(tmp_path, {"train": 1, "val": 1}, noisy_counts={"train": 1})


def test_source_lists_splits_in_order_and_reads_exact_pixels(tmp_path):
    write_square_source(tmp_path, 3, {"train": 2, "val": 1}, SMALL_IMAGES)

    images = source_images(tmp_path)
    assert [(split, path.name) for split, path in images] == [
        ("train", "square_00000000.png"),
        ("train", "square_00000001.png"),
        ("val", "square_00000002.png"),
    ]
    np.testing.assert_array_equal(
        read_image(images[2][1]), _family(2)[0].tree_image.image
    )


def test_colour_images_are_rejected(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")

    with pytest.raises(ValueError, match="grayscale"):
        read_image(tmp_path / "rgb.png")
