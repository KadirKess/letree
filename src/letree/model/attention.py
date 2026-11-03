"""Exact graph-local parent attention calculations."""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)


@dataclass(frozen=True)
class ParentAttentionOutput:
    parent_message: Tensor
    query: Tensor
    key: Tensor
    log_normalizers: Tensor


def readable_graph_attention(
    query: Tensor,
    key: Tensor,
    hidden_state: Tensor,
    graph_ptr: Tensor,
) -> ParentAttentionOutput:
    """Compute all-node parent attention separately for each packed graph."""

    # Scaling the queries once is cheaper than scaling every n by n score block
    scaled_query = query / sqrt(query.shape[1])
    parent_messages = []
    log_normalizers = []

    # Score candidates and form expected parent states within each image. Reading
    # the bounds once avoids one device synchronization per graph.
    bounds = graph_ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        graph_scores = scaled_query[start:stop] @ key[start:stop].T
        graph_probabilities = F.softmax(graph_scores, dim=1)

        parent_messages.append(graph_probabilities @ hidden_state[start:stop])
        log_normalizers.append(torch.logsumexp(graph_scores, dim=1))

    return ParentAttentionOutput(
        parent_message=torch.cat(parent_messages, dim=0),
        query=query,
        key=key,
        log_normalizers=torch.cat(log_normalizers, dim=0),
    )


DEFAULT_KERNEL_SHARED_MEMORY = 147_968
SMALL_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}

compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


def same_graph_block_mask(graph_ptr: Tensor) -> BlockMask:
    node_count = int(graph_ptr[-1])
    graph_sizes = graph_ptr[1:] - graph_ptr[:-1]
    graph_ids = torch.repeat_interleave(
        torch.arange(graph_sizes.numel(), device=graph_ptr.device), graph_sizes
    )

    def same_graph(batch, head, query_index, key_index):
        return graph_ids[query_index] == graph_ids[key_index]

    return create_block_mask(
        same_graph, None, None, node_count, node_count, device=graph_ptr.device
    )


def flex_kernel_options(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    properties = torch.cuda.get_device_properties(device)
    if properties.shared_memory_per_block_optin < DEFAULT_KERNEL_SHARED_MEMORY:
        return SMALL_KERNEL_OPTIONS
    return {}


def flex_graph_attention(
    query: Tensor,
    key: Tensor,
    hidden_state: Tensor,
    block_mask: BlockMask,
) -> ParentAttentionOutput:
    """Dense attention streamed in tiles, so large graphs never store n by n scores."""
    attend = compiled_flex_attention if query.is_cuda else flex_attention
    parent_message, aux = attend(
        query[None, None],
        key[None, None],
        hidden_state[None, None],
        block_mask=block_mask,
        scale=1 / sqrt(query.shape[1]),
        kernel_options=flex_kernel_options(query.device),
        return_aux=AuxRequest(lse=True),
    )
    return ParentAttentionOutput(
        parent_message=parent_message[0, 0],
        query=query,
        key=key,
        log_normalizers=aux.lse[0, 0],
    )
