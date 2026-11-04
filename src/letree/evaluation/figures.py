"""Figures of training curves and of chosen samples."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

LOSS_NAMES = ("observed_parent", "corrective_parent", "action")
DECODED_SOURCES = ("observed", "model", "oracle")


def zone_boundaries(labels: NDArray[np.integer]) -> NDArray[np.bool_]:
    """Pixels whose right or lower neighbour lies in another zone."""
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


def _show(axis, values, title, **style) -> None:
    image = axis.imshow(values, interpolation="nearest", **style)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    if style.get("cmap") != "gray":
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _outline(axis, image, labels, title) -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    overlay = np.zeros((*labels.shape, 4))
    overlay[zone_boundaries(labels)] = (1.0, 0.2, 0.2, 0.9)
    axis.imshow(overlay, interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def _failed(axis, source, failure) -> None:
    axis.text(0.5, 0.5, f"{source}\ninvalid\n{failure}", ha="center", va="center")
    axis.set_xticks([])
    axis.set_yticks([])


def sample_figure(sample: dict) -> Figure:
    figure, axes = plt.subplots(2, 5, figsize=(17, 7))
    clean_depth = sample["clean_depth"]
    shared = [clean_depth] + [
        sample[source]["depth"]
        for source in ("model", "oracle")
        if sample[source]["failure"] is None
    ]
    # The observed tree of a noisy image is much deeper, so it keeps its own scale
    depth_scale = {"cmap": "viridis", "vmin": 0, "vmax": max(d.max() for d in shared)}

    _show(axes[0, 0], clean_depth, "clean depth", **depth_scale)
    for column, source in enumerate(DECODED_SOURCES, start=1):
        decoded = sample[source]
        if decoded["failure"] is not None:
            _failed(axes[0, column], source, decoded["failure"])
            continue
        style = {"cmap": "viridis"} if source == "observed" else depth_scale
        _show(axes[0, column], decoded["depth"], f"{source} depth", **style)

    model = sample["model"]
    if model["failure"] is None:
        _show(
            axes[0, 4],
            model["depth_difference"],
            f"model depth error, MAE {model['depth_difference'].mean():.2f}",
            cmap="magma",
        )
    else:
        _failed(axes[0, 4], "model", model["failure"])

    _outline(axes[1, 0], sample["clean_image"], sample["clean_labels"], "clean zones")
    _outline(axes[1, 1], sample["image"], sample["observed_labels"], "observed zones")
    for column, source in ((2, "model"), (3, "oracle")):
        if sample[source]["failure"] is None:
            _outline(
                axes[1, column],
                sample["image"],
                sample[source]["groups"],
                f"{source} groups",
            )
        else:
            _failed(axes[1, column], source, sample[source]["failure"])
    if model["failure"] is None:
        _show(
            axes[1, 4],
            model["misassigned"],
            f"model misassigned, {model['misassigned'].mean():.1%}",
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )
    else:
        _failed(axes[1, 4], "model", model["failure"])

    figure.suptitle(f"{sample['key']}  {sample['corruption']}")
    figure.tight_layout()
    return figure


def training_figure(history: list[dict]) -> Figure:
    epochs = [entry["epoch"] for entry in history]
    validation = [entry["validation"] for entry in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axis = axes[0, 0]
    for name, colour in zip(LOSS_NAMES, ("C0", "C1", "C2")):
        axis.plot(epochs, [entry["train"][name] for entry in history], colour)
        axis.plot(
            epochs,
            [entry[f"loss/{name}"] for entry in validation],
            colour,
            linestyle="--",
        )
        axis.plot([], [], colour, label=name)
    axis.plot([], [], "k", label="train")
    axis.plot([], [], "k--", label="validation")
    axis.set_title("losses")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    suffix = "/joint_correction_accuracy/image_mean"
    groups = [name[: -len(suffix)] for name in validation[0] if name.endswith(suffix)]
    for group in groups:
        axis.plot(epochs, [entry[group + suffix] for entry in validation], label=group)
    axis.set_title("validation joint accuracy, image mean")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.semilogy(epochs, [entry["learning_rate"] for entry in history])
    axis.set_title("learning rate")

    axis = axes[1, 1]
    for name in ("query", "key"):
        lengths = [entry.get(f"max_length/{name}") for entry in validation]
        if any(length is not None for length in lengths):
            axis.plot(epochs, lengths, label=name)
    axis.set_title("largest query and key length, validation")
    axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("epoch")
    figure.tight_layout()
    return figure
