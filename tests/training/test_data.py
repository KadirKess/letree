import pytest

from letree.component_tree import ComponentTreeKind
from letree.dataset.build import generate_dataset
from letree.generation import (
    AdditiveGaussianNoise,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    SquareImageConfig,
)
from letree.training.data import ObservationDataset, make_loader

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("observations") / "data"
    generate_dataset(
        root,
        7,
        {"train": 2, "val": 0},
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        (SaltAndPepperNoise(probability=0.05),),
        2,
    )
    return root


@pytest.fixture(scope="module")
def mixed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed") / "data"
    generate_dataset(
        root,
        7,
        {"train": 2},
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        (
            SaltAndPepperNoise(probability=0.05),
            AdditiveGaussianNoise(standard_deviation=10.0),
            SpeckleNoise(),
            PoissonNoise(),
        ),
        1,
    )
    return root


def test_main_run_serves_clean_then_noisy_observations(dataset_root) -> None:
    dataset = ObservationDataset(dataset_root, "train", include_noisy=True)

    assert dataset.items == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    batch = next(iter(make_loader(dataset, 6, shuffle=False, seed=0)))
    assert batch.is_clean.tolist() == [True, False, False, True, False, False]
    assert batch.num_graphs == 6


def test_clean_only_control_serves_no_noisy_observation(dataset_root) -> None:
    dataset = ObservationDataset(dataset_root, "train", include_noisy=False)

    assert dataset.items == [(0, 0), (1, 0)]
    assert all(bool(dataset[i].is_clean) for i in range(len(dataset)))


def test_empty_split_is_rejected(dataset_root) -> None:
    with pytest.raises(ValueError, match="no families"):
        ObservationDataset(dataset_root, "val", include_noisy=True)


def test_seeded_shuffle_is_reproducible(dataset_root) -> None:
    dataset = ObservationDataset(dataset_root, "train", include_noisy=True)

    def order():
        loader = make_loader(dataset, 1, shuffle=True, seed=4)
        return [int(batch.num_nodes) for batch in loader]

    assert order() == order()


def test_mixed_dataset_serves_every_noise_kind_of_each_family(mixed_root) -> None:
    dataset = ObservationDataset(mixed_root, "train", include_noisy=True)

    assert dataset.items == [(i, p) for i in range(2) for p in range(5)]
    batch = next(iter(make_loader(dataset, 10, shuffle=False, seed=0)))
    kinds = ["clean", "salt_and_pepper", "additive_gaussian", "speckle", "poisson"]
    assert batch.corruption == kinds * 2
    assert batch.is_clean.tolist() == [True, False, False, False, False] * 2


def test_clean_only_control_on_a_mixed_dataset(mixed_root) -> None:
    dataset = ObservationDataset(mixed_root, "train", include_noisy=False)

    assert dataset.items == [(0, 0), (1, 0)]
    assert [dataset[i].corruption for i in range(2)] == ["clean", "clean"]
