import numpy as np
import pytest

from letree.component_tree import ComponentTreeKind
from letree.dataset import NpzDataset
from letree.dataset.build import build_sample_family, generate_dataset
from letree.generation import (
    AdditiveGaussianNoise,
    GenerationStage,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    SquareImageConfig,
    corrupt_image,
    derive_rng,
    generate_square_image,
)

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)
NOISE = SaltAndPepperNoise(probability=0.05)
GAUSSIAN = AdditiveGaussianNoise(standard_deviation=10.0)
ALL_NOISES = (NOISE, GAUSSIAN, SpeckleNoise(), PoissonNoise())


def _family(sample_index: int, base_seed: int = 3, noises=(NOISE,), realizations=2):
    return build_sample_family(
        base_seed,
        sample_index,
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        noises,
        realizations,
    )


def _images(family):
    return [family.clean.observed.image] + [
        o.sample.observed.image for o in family.noisy
    ]


def test_family_is_reproducible_from_its_seeds() -> None:
    first, second = _family(5), _family(5)

    for a, b in zip(_images(first), _images(second), strict=True):
        np.testing.assert_array_equal(a, b)


def test_family_holds_clean_and_distinct_realizations() -> None:
    family = _family(5)

    assert [o.realization_index for o in family.noisy] == [0, 1]
    assert all(o.noise_config == NOISE for o in family.noisy)
    for observation in family.noisy:
        np.testing.assert_array_equal(
            observation.sample.clean.image, family.clean.clean.image
        )
    assert not np.array_equal(*[o.sample.observed.image for o in family.noisy])


def test_noisy_observations_follow_noise_then_realization_order() -> None:
    family = _family(5, noises=(NOISE, GAUSSIAN))

    assert [(o.noise_config, o.realization_index) for o in family.noisy] == [
        (NOISE, 0),
        (NOISE, 1),
        (GAUSSIAN, 0),
        (GAUSSIAN, 1),
    ]


@pytest.mark.parametrize(
    ("noise", "stage"),
    [
        (NOISE, GenerationStage.SALT_AND_PEPPER_NOISE),
        (GAUSSIAN, GenerationStage.ADDITIVE_GAUSSIAN_NOISE),
        (SpeckleNoise(), GenerationStage.SPECKLE_NOISE),
        (PoissonNoise(), GenerationStage.POISSON_NOISE),
    ],
)
def test_each_observation_draws_from_its_kind_stream(noise, stage) -> None:
    # Rebuilt from the documented streams, not from the family builder, so the
    # single-noise datasets generated before mixing stay reproducible
    family = _family(5, noises=ALL_NOISES)
    clean = generate_square_image(
        SMALL_IMAGES, derive_rng(3, 5, GenerationStage.SQUARE_IMAGE)
    )

    observations = [o for o in family.noisy if o.noise_config == noise]
    assert [o.realization_index for o in observations] == [0, 1]
    for observation in observations:
        rng = derive_rng(3, 5, stage, realization_index=observation.realization_index)
        np.testing.assert_array_equal(
            observation.sample.observed.image, corrupt_image(clean, noise, rng)
        )


def test_adding_a_noise_kind_leaves_other_kinds_unchanged() -> None:
    alone = _family(5, noises=(GAUSSIAN,))
    mixed = _family(5, noises=(NOISE, GAUSSIAN))

    mixed_gaussian = [o for o in mixed.noisy if o.noise_config == GAUSSIAN]
    for a, b in zip(alone.noisy, mixed_gaussian, strict=True):
        np.testing.assert_array_equal(a.sample.observed.image, b.sample.observed.image)


def test_two_configurations_of_one_kind_are_rejected() -> None:
    # They would read the same stream, so their noise would be correlated
    with pytest.raises(ValueError, match="kind"):
        _family(5, noises=(GAUSSIAN, AdditiveGaussianNoise(standard_deviation=5.0)))


@pytest.mark.parametrize("workers", [1, 2])
def test_splits_take_disjoint_index_ranges(tmp_path, workers) -> None:
    counts = {"train": 3, "val": 2, "test": 1}
    generate_dataset(
        tmp_path / "data",
        3,
        counts,
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        (NOISE,),
        2,
        workers,
    )

    names = {
        split: sorted(p.name for p in (tmp_path / "data" / split).glob("*.npz"))
        for split in counts
    }
    assert names == {
        "train": [f"sample_{i:08d}.npz" for i in range(3)],
        "val": [f"sample_{i:08d}.npz" for i in range(3, 5)],
        "test": ["sample_00000005.npz"],
    }
    loaded = NpzDataset.open(tmp_path / "data").load("val", 3)
    np.testing.assert_array_equal(
        loaded.clean.observed.image, _family(3).clean.observed.image
    )


def test_mixed_noise_dataset_round_trips(tmp_path) -> None:
    generate_dataset(
        tmp_path / "data",
        3,
        {"train": 2},
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        ALL_NOISES,
        2,
    )

    loaded = NpzDataset.open(tmp_path / "data").load("train", 1)
    expected = _family(1, noises=ALL_NOISES)
    assert [(o.noise_config, o.realization_index) for o in loaded.noisy] == [
        (o.noise_config, o.realization_index) for o in expected.noisy
    ]
    for a, b in zip(_images(loaded), _images(expected), strict=True):
        np.testing.assert_array_equal(a, b)
