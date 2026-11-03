"""Argmax accuracies and trivial baselines per image."""

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from letree.generation.targets import CorrectionAction
from letree.model import Output

IMAGE_QUANTITIES = (
    "observed_parent_accuracy",
    "corrective_parent_accuracy",
    "action_accuracy",
    "joint_correction_accuracy",
    "copy_observed_baseline",
    "always_merge_baseline",
)


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
    """One value per image for each quantity, plus node counts and tie counts."""
    observed, corrective = output.observed[-1], output.corrective[-1]
    observed_predictions, observed_tied = graph_local_argmax(
        observed.query, observed.key, data.ptr
    )
    corrective_predictions, corrective_tied = graph_local_argmax(
        corrective.query, corrective.key, data.ptr
    )
    corrective_correct = corrective_predictions == data.corrective_parents
    action_correct = output.actions[-1].argmax(dim=1) == data.correction_actions

    def per_image(correct: Tensor) -> Tensor:
        return scatter(
            correct.float(), data.batch, reduce="mean", dim_size=data.num_graphs
        )

    def count(mask: Tensor) -> Tensor:
        return scatter(mask.long(), data.batch, reduce="sum", dim_size=data.num_graphs)

    return {
        "observed_parent_accuracy": per_image(
            observed_predictions == data.observed_parents
        ),
        "corrective_parent_accuracy": per_image(corrective_correct),
        "action_accuracy": per_image(action_correct),
        # A node decodes correctly only when both its pointer and action are right
        "joint_correction_accuracy": per_image(corrective_correct & action_correct),
        "copy_observed_baseline": per_image(
            data.observed_parents == data.corrective_parents
        ),
        "always_merge_baseline": per_image(
            data.correction_actions == CorrectionAction.MERGE
        ),
        "observed_ties": count(observed_tied),
        "corrective_ties": count(corrective_tied),
        "node_count": data.ptr.diff(),
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
    node_count = values["node_count"].double()

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
        weights = node_count[mask]
        for name in IMAGE_QUANTITIES:
            value = values[name][mask].double()
            summary[f"{group}/{name}/image_mean"] = value.mean().item()
            # A per-image accuracy times its node count is its number of correct nodes
            summary[f"{group}/{name}/pooled"] = (
                (value * weights).sum() / weights.sum()
            ).item()
        summary[f"{group}/images"] = int(mask.sum())
        summary[f"{group}/nodes"] = int(weights.sum())
        summary[f"{group}/observed_ties"] = int(values["observed_ties"][mask].sum())
        summary[f"{group}/corrective_ties"] = int(values["corrective_ties"][mask].sum())
    return summary
