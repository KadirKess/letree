import pytest

from letree.component_tree import ComponentTreeKind
from letree.data.families import build_family, generate_dataset
from letree.data.sources import write_square_source
from letree.generation import (
    GenerationStage,
    SquareImageConfig,
    derive_rng,
    generate_square_image,
)

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)


@pytest.fixture(scope="session")
def square_family():
    """The family the square pipeline builds for one image index."""

    def build(
        base_seed, sample_index, noises, noisy_count, kind=ComponentTreeKind.MAX_TREE
    ):
        rng = derive_rng(base_seed, sample_index, GenerationStage.SQUARE_IMAGE)
        image = generate_square_image(SMALL_IMAGES, rng)
        return build_family(image, base_seed, sample_index, kind, noises, noisy_count)

    return build


@pytest.fixture(scope="session")
def make_dataset(tmp_path_factory):
    """Write a small square source, then a dataset with noisy_count in every split."""

    def make(
        counts,
        noises,
        noisy_count,
        base_seed=3,
        kind=ComponentTreeKind.MAX_TREE,
        workers=1,
    ):
        directory = tmp_path_factory.mktemp("dataset")
        write_square_source(directory / "source", base_seed, counts, SMALL_IMAGES)
        generate_dataset(
            directory / "data",
            directory / "source",
            base_seed,
            kind,
            noises,
            {split: noisy_count for split in counts},
            workers,
        )
        return directory / "data"

    return make
