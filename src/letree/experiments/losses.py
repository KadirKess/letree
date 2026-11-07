"""Graph-local task losses with image-weighted reductions."""

from math import sqrt

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from letree.attention import flex_target_set_log_z, readable_target_set_log_z
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


def set_pointer_cross_entropy(step: Step, data: Batch, attention: str) -> Tensor:
    target_set_log_z = (
        flex_target_set_log_z if attention == "flex" else readable_target_set_log_z
    )
    return step.log_z - target_set_log_z(
        step.query, step.key, data.zone, data.parent_zone, data.ptr
    )


def zone_pointer_cross_entropy(
    set_pointer: Tensor, data: Batch
) -> tuple[Tensor, Tensor]:
    """Minus the log of a zone's mean parent-zone mass, and the zone's image."""
    # The soft vote reads this mean, while the node loss trains the mean of
    # its logs. Zone ids are local to their graph, so pair them with the graph
    zone_count = int(data.zone.max()) + 1
    key = data.batch * zone_count + data.zone
    zone_keys, zone_of_node = torch.unique(key, return_inverse=True)
    log_mass = -set_pointer
    # Shifting by the zone maximum keeps the exponentials finite
    largest = scatter(
        log_mass.detach(), zone_of_node, dim=0, dim_size=zone_keys.numel(), reduce="max"
    )
    sums = scatter(
        torch.exp(log_mass - largest[zone_of_node]),
        zone_of_node,
        dim=0,
        dim_size=zone_keys.numel(),
    )
    sizes = torch.bincount(zone_of_node, minlength=zone_keys.numel())
    log_mean = largest + torch.log(sums) - torch.log(sizes.to(sums.dtype))
    return -log_mean, zone_keys // zone_count


def contraction_targets(data: Batch) -> Tensor:
    # An edge contracts when both ends belong to the same clean zone. Edges never
    # cross graphs, so the local zone ids compare directly
    source, destination = data.rag_edge_index
    return (data.zone[source] == data.zone[destination]).float()


def image_mean(values: Tensor, data: Batch, image: Tensor | None = None) -> Tensor:
    """Mean per image, then over images, of node values or of values per `image`."""
    image = data.batch if image is None else image
    # Every image weighs the same in the batch, whatever its node or edge count
    per_image = scatter(values, image, dim=0, reduce="mean", dim_size=data.num_graphs)
    present = torch.bincount(image, minlength=data.num_graphs) > 0
    return per_image[present].mean()


def observed_losses(output: Output, data: Batch) -> dict[str, Tensor]:
    """Observed loss at the last observed step."""
    observed = parent_cross_entropy(output.observed[-1], data.observed_parents, data)
    return {"observed_parent": image_mean(observed, data)}


def correction_losses(
    output: Output,
    data: Batch,
    attention: str = "dense",
    zone_pointer: bool = False,
    separating_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Observed loss at the last observed step, the rest at the last corrective one."""
    set_pointer = set_pointer_cross_entropy(output.corrective[-1], data, attention)
    contraction_logits, increment_logits = output.contraction[-1], output.increment[-1]
    targets = contraction_targets(data).to(contraction_logits.dtype)
    contraction = F.binary_cross_entropy_with_logits(
        contraction_logits,
        targets,
        # A false merge costs more than a false split, so separating edges weigh more
        weight=1 + (separating_weight - 1) * (1 - targets),
        reduction="none",
    )
    increment = F.binary_cross_entropy_with_logits(
        increment_logits, data.increment.to(increment_logits.dtype), reduction="none"
    )
    edge_image = data.batch[data.rag_edge_index[0]]
    losses = observed_losses(output, data) | {
        "set_pointer": image_mean(set_pointer, data),
        "contraction": image_mean(contraction, data, edge_image),
        "increment": image_mean(increment, data),
    }
    if zone_pointer:
        values, zone_image = zone_pointer_cross_entropy(set_pointer, data)
        losses["zone_pointer"] = image_mean(values, data, zone_image)
    return losses
