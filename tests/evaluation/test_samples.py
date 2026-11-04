import numpy as np

from letree.evaluation.checkpoint import load_run_model
from letree.evaluation.samples import choose_samples, sample_maps, split_samples


def _samples(families, kinds):
    return [
        (f"sample_{family:08d}.npz", position, kind)
        for family in range(families)
        for position, kind in enumerate(kinds)
    ]


def test_choice_depends_only_on_the_seed():
    samples = _samples(20, ["clean", "gauss", "speckle"])
    shuffled = samples[::-1]
    assert choose_samples(samples, 2, 5) == choose_samples(shuffled, 2, 5)
    assert choose_samples(samples, 2, 5) != choose_samples(samples, 2, 6)


def test_choice_takes_each_group_from_distinct_families():
    keys = choose_samples(_samples(20, ["clean", "gauss", "speckle"]), 3, 0)
    assert len(keys) == 9
    families = [key.split("/")[0] for key in keys]
    assert len(set(families)) == 9
    positions = [int(key.split("/")[1]) for key in keys]
    assert sorted(positions) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_maps_of_chosen_samples(trained_run):
    root, run = trained_run
    samples = split_samples(root, "test")
    assert len(samples) == 9
    keys = choose_samples(samples, 1, 0)
    model, settings, device = load_run_model(run)

    maps = sample_maps(root, "test", keys, model, settings, device)

    assert [sample["key"] for sample in maps] == keys
    for sample in maps:
        oracle = sample["oracle"]
        assert oracle["failure"] is None
        assert sample["clean_depth"].shape == (16, 16)
        if sample["corruption"] == "clean":
            np.testing.assert_array_equal(oracle["depth"], sample["clean_depth"])
            assert not oracle["misassigned"].any()
        model_maps = sample["model"]
        if model_maps["failure"] is None:
            np.testing.assert_array_equal(
                model_maps["depth_difference"],
                np.abs(model_maps["depth"] - sample["clean_depth"]),
            )
