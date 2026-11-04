import pytest

from letree.experiments.train import RunSettings, fit
from letree.generation import AdditiveGaussianNoise, SaltAndPepperNoise

NOISES = (SaltAndPepperNoise(probability=0.1), AdditiveGaussianNoise(10.0))


@pytest.fixture(scope="session")
def trained_run(make_dataset, tmp_path_factory):
    """A tiny dataset and a one-epoch run trained on it."""
    root = make_dataset({"train": 2, "val": 1, "test": 3}, NOISES, 2, base_seed=4)
    run = tmp_path_factory.mktemp("run")
    settings = RunSettings(
        dataset_root=str(root),
        max_epochs=1,
        width=16,
        observed_steps=2,
        correction_steps=1,
        batch_size=4,
    )
    fit(settings, run)
    return root, run
