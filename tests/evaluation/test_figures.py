import json

import matplotlib.pyplot as plt
import numpy as np

from letree.evaluation.checkpoint import load_run_model
from letree.evaluation.figures import sample_figure, training_figure, zone_boundaries
from letree.evaluation.samples import choose_samples, sample_maps, split_samples


def test_zone_boundaries_mark_the_first_pixel_of_each_change():
    labels = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 2]])
    np.testing.assert_array_equal(
        zone_boundaries(labels),
        [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
    )


def test_figures_draw_a_run_and_its_samples(trained_run):
    root, run = trained_run
    history = json.loads((run / "history.json").read_text())
    plt.close(training_figure(history))

    model, settings, device = load_run_model(run)
    keys = choose_samples(split_samples(root, "test"), 1, 0)
    for sample in sample_maps(root, "test", keys, model, settings, device):
        figure = sample_figure(sample)
        assert len(figure.axes) >= 10
        plt.close(figure)


def test_an_invalid_model_decode_still_draws():
    labels = np.array([[0, 1], [1, 1]])
    depth = np.array([[0, 1], [1, 1]])
    decoded = {
        "failure": None,
        "groups": labels,
        "depth": depth,
        "misassigned": np.zeros((2, 2), dtype=bool),
        "depth_difference": np.zeros((2, 2), dtype=np.int64),
    }
    sample = {
        "key": "sample_00000000/1",
        "corruption": "gauss",
        "clean_image": np.zeros((2, 2), dtype=np.uint8),
        "image": np.zeros((2, 2), dtype=np.uint8),
        "clean_labels": labels,
        "clean_depth": depth,
        "observed_labels": labels,
        "observed": decoded,
        "oracle": decoded,
        "model": {"failure": "root_count"},
    }
    plt.close(sample_figure(sample))
