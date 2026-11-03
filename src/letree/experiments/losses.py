"""Graph-local task losses with image-weighted reductions."""

from math import sqrt

from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from letree.model import Output, Step


def parent_cross_entropy(step: Step, local_targets: Tensor, data: Batch) -> Tensor:
    """Per-node cross-entropy as log Z minus the target score.

    This equals ordinary cross-entropy over each graph's score row without
    reading a stored score matrix, so a streaming attention backend can supply
    log Z instead.
    """
    # Targets are local to their graph, so shift them to rows of the packed batch
    targets = local_targets + data.ptr[data.batch]
    target_scores = (step.query * step.key[targets]).sum(1) / sqrt(step.query.shape[1])
    return step.log_z - target_scores


def image_mean(node_values: Tensor, data: Batch) -> Tensor:
    # Every image weighs the same in the batch, whatever its node count
    per_image = scatter(
        node_values, data.batch, dim=0, reduce="mean", dim_size=data.num_graphs
    )
    return per_image.mean()


def correction_losses(output: Output, data: Batch) -> dict[str, Tensor]:
    """Observed loss at the last observed step, corrective losses at the last index."""
    observed = parent_cross_entropy(output.observed[-1], data.observed_parents, data)
    corrective = parent_cross_entropy(
        output.corrective[-1], data.corrective_parents, data
    )
    action = F.cross_entropy(
        output.actions[-1], data.correction_actions, reduction="none"
    )
    return {
        "observed_parent": image_mean(observed, data),
        "corrective_parent": image_mean(corrective, data),
        "action": image_mean(action, data),
    }
