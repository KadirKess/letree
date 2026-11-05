"""Argmax and sign accuracies per image."""

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from letree.experiments.losses import contraction_targets
from letree.model import Output

IMAGE_QUANTITIES = (
    "observed_parent_accuracy",
    "set_pointer_accuracy",
    "increment_accuracy",
    "joint_accuracy",
    "edge_accuracy",
)
# Edge accuracy is a share of edges, so it is pooled by edges and not nodes
EDGE_QUANTITIES = {"edge_accuracy"}


@torch.no_grad()
def graph_local_argmax(
    query: Tensor, key: Tensor, graph_ptr: Tensor
) -> tuple[Tensor, Tensor]:
    """Local argmax parents and a mask of nodes whose maximum is tied.

    The argmax takes the first maximal candidate. The score scale is skipped
    because it does not change the argmax.
    """
    predictions, tied = [], []
    bounds = graph_ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        scores = query[start:stop] @ key[start:stop].T
        best = scores.max(dim=1, keepdim=True).values
        predictions.append(scores.argmax(dim=1))
        tied.append((scores == best).sum(dim=1) > 1)
    return torch.cat(predictions), torch.cat(tied)


def batch_image_metrics(output: Output, data: Batch) -> dict[str, Tensor]:
    """One value per image for each quantity, plus node, edge and tie counts."""
    observed, corrective = output.observed[-1], output.corrective[-1]
    observed_predictions, observed_tied = graph_local_argmax(
        observed.query, observed.key, data.ptr
    )
    corrective_predictions, corrective_tied = graph_local_argmax(
        corrective.query, corrective.key, data.ptr
    )
    # Predictions are local to their graph, so shift them to rows of the batch
    pointed = corrective_predictions + data.ptr[data.batch]
    pointer_correct = data.zone[pointed] == data.parent_zone
    increment_correct = (output.increment[-1] > 0) == data.increment.bool()
    edge_correct = (output.contraction[-1] > 0) == contraction_targets(data).bool()
    edge_image = data.batch[data.rag_edge_index[0]]

    def per_image(correct: Tensor, image: Tensor = data.batch) -> Tensor:
        return scatter(correct.float(), image, reduce="mean", dim_size=data.num_graphs)

    def count(mask: Tensor) -> Tensor:
        return scatter(mask.long(), data.batch, reduce="sum", dim_size=data.num_graphs)

    return {
        "observed_parent_accuracy": per_image(
            observed_predictions == data.observed_parents
        ),
        "set_pointer_accuracy": per_image(pointer_correct),
        "increment_accuracy": per_image(increment_correct),
        # A node decodes correctly only when its pointer and increment are right
        "joint_accuracy": per_image(pointer_correct & increment_correct),
        "edge_accuracy": per_image(edge_correct, edge_image),
        "observed_ties": count(observed_tied),
        "corrective_ties": count(corrective_tied),
        "node_count": data.ptr.diff(),
        "edge_count": torch.bincount(edge_image, minlength=data.num_graphs),
    }


def summarize_image_metrics(per_image: dict[str, list]) -> dict[str, float]:
    """Image mean and node-pooled value of each quantity per group of images.

    The groups are all images, clean ones, noisy ones and each noise kind, read
    from the corruption names that come with the per-image values.
    """
    corruption = [name for batch in per_image["corruption"] for name in batch]
    values = {
        name: torch.cat(batches)
        for name, batches in per_image.items()
        if name != "corruption"
    }
    counts = {
        "node": values["node_count"].double(),
        "edge": values["edge_count"].double(),
    }

    def images_of(*kinds):
        return torch.tensor([name in kinds for name in corruption])

    noise_kinds = sorted(set(corruption) - {"clean"})
    groups = [
        ("all", torch.ones(len(corruption), dtype=torch.bool)),
        ("clean", images_of("clean")),
        ("noisy", images_of(*noise_kinds)),
        *[(kind, images_of(kind)) for kind in noise_kinds],
    ]
    summary = {}
    for group, mask in groups:
        if not mask.any():
            continue
        for name in IMAGE_QUANTITIES:
            weights = counts["edge" if name in EDGE_QUANTITIES else "node"][mask]
            # Images without edges have no edge accuracy, so they are left out
            present = weights > 0
            value = values[name][mask].double()[present]
            summary[f"{group}/{name}/image_mean"] = value.mean().item()
            # A per-image accuracy times its count is its number of correct items
            summary[f"{group}/{name}/pooled"] = (
                (value * weights[present]).sum() / weights.sum()
            ).item()
        summary[f"{group}/images"] = int(mask.sum())
        summary[f"{group}/nodes"] = int(counts["node"][mask].sum())
        summary[f"{group}/edges"] = int(counts["edge"][mask].sum())
        summary[f"{group}/observed_ties"] = int(values["observed_ties"][mask].sum())
        summary[f"{group}/corrective_ties"] = int(values["corrective_ties"][mask].sum())
    return summary
