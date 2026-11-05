"""Figures of training curves and of chosen samples."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

LOSS_NAMES = ("observed_parent", "set_pointer", "contraction", "increment")
DECODED_SOURCES = ("observed", "model_soft", "model_hard", "oracle")
MODEL_SOURCES = ("model_soft", "model_hard")


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


def _failed(axis, source, decoded) -> None:
    text = f"{source}\ninvalid\n{decoded['failure']}\n{decoded['root_claims']} roots"
    axis.text(0.5, 0.5, text, ha="center", va="center")
    axis.set_xticks([])
    axis.set_yticks([])


def sample_figure(sample: dict) -> Figure:
    columns = 1 + len(DECODED_SOURCES) + len(MODEL_SOURCES)
    figure, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 7))
    clean_depth = sample["clean_depth"]
    shared = [clean_depth] + [
        sample[source]["depth"]
        for source in DECODED_SOURCES
        if source != "observed" and not sample[source]["failure"]
    ]
    # The observed tree of a noisy image is much deeper, so it keeps its own scale
    depth_scale = {"cmap": "viridis", "vmin": 0, "vmax": max(d.max() for d in shared)}

    _show(axes[0, 0], clean_depth, "clean depth", **depth_scale)
    _outline(axes[1, 0], sample["clean_image"], sample["clean_labels"], "clean zones")
    for column, source in enumerate(DECODED_SOURCES, start=1):
        decoded = sample[source]
        if decoded["failure"]:
            _failed(axes[0, column], source, decoded)
            _failed(axes[1, column], source, decoded)
            continue
        style = {"cmap": "viridis"} if source == "observed" else depth_scale
        _show(axes[0, column], decoded["depth"], f"{source} depth", **style)
        _outline(
            axes[1, column],
            sample["image"],
            decoded["group_map"],
            f"{source} groups, {decoded['groups']}",
        )

    for column, source in enumerate(MODEL_SOURCES, start=1 + len(DECODED_SOURCES)):
        decoded = sample[source]
        if decoded["failure"]:
            _failed(axes[0, column], source, decoded)
            _failed(axes[1, column], source, decoded)
            continue
        _show(
            axes[0, column],
            decoded["depth_difference"],
            f"{source} depth error, MAE {decoded['depth_difference'].mean():.2f}",
            cmap="magma",
        )
        _show(
            axes[1, column],
            decoded["misassigned"],
            f"{source} misassigned, {decoded['misassigned'].mean():.1%}",
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )

    figure.suptitle(f"{sample['key']}  {sample['corruption']}")
    figure.tight_layout()
    return figure


def training_figure(history: list[dict]) -> Figure:
    epochs = [entry["epoch"] for entry in history]
    validation = [entry["validation"] for entry in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axis = axes[0, 0]
    for name, colour in zip(LOSS_NAMES, ("C0", "C1", "C2", "C3")):
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
    suffix = "/joint_accuracy/image_mean"
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
