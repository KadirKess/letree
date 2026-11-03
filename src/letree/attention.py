"""Graph-local parent attention, as a readable loop and as FlexAttention."""

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


def readable_graph_attention(
    query: Tensor, key: Tensor, hidden: Tensor, graph_ptr: Tensor
) -> tuple[Tensor, Tensor]:
    """Attention computed separately for each packed graph."""
    scaled_query = query / sqrt(query.shape[1])
    messages, log_z = [], []

    bounds = graph_ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        scores = scaled_query[start:stop] @ key[start:stop].T
        messages.append(F.softmax(scores, dim=1) @ hidden[start:stop])
        log_z.append(torch.logsumexp(scores, dim=1))
    return torch.cat(messages), torch.cat(log_z)


# Bytes of shared memory the default kernel asked for in Triton's error on an L4.
# Tile options are under FlexKernelOptions in
# https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
DEFAULT_KERNEL_SHARED_MEMORY = 147_968
SMALL_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}

compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


def round_to_tf32(values: Tensor) -> Tensor:
    """Float32 values rounded to the 10 mantissa bits a TF32 matmul reads.

    The GPU truncates float32 inputs to TF32, which biases every score towards
    zero, while the loss computes its target score in full float32. Rounding
    first makes both read the same values, whose products are exact in float32.
    The gradient passes through unchanged.
    """
    bits = values.view(torch.int32)
    rounded = ((bits + 0x1000) & ~0x1FFF).view(torch.float32)
    return values + (rounded - values).detach()


def same_graph_block_mask(graph_ptr: Tensor) -> BlockMask:
    node_count = int(graph_ptr[-1])
    graph_ids = torch.repeat_interleave(
        torch.arange(graph_ptr.numel() - 1, device=graph_ptr.device), graph_ptr.diff()
    )

    def same_graph(batch, head, query_index, key_index):
        return graph_ids[query_index] == graph_ids[key_index]

    # Uncompiled, the mask function runs on every node pair and needs n by n
    # memory, which runs out on a GPU near 40,000 nodes
    return create_block_mask(
        same_graph,
        None,
        None,
        node_count,
        node_count,
        device=graph_ptr.device,
        _compile=graph_ptr.is_cuda,
    )


def flex_kernel_options(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    properties = torch.cuda.get_device_properties(device)
    if properties.shared_memory_per_block_optin < DEFAULT_KERNEL_SHARED_MEMORY:
        return SMALL_KERNEL_OPTIONS
    return {}


def flex_graph_attention(
    query: Tensor, key: Tensor, hidden: Tensor, block_mask: BlockMask
) -> tuple[Tensor, Tensor]:
    """Dense attention streamed in tiles, so large graphs never store n by n scores."""
    attend = compiled_flex_attention if query.is_cuda else flex_attention
    message, aux = attend(
        query[None, None],
        key[None, None],
        hidden[None, None],
        block_mask=block_mask,
        scale=1 / sqrt(query.shape[1]),
        kernel_options=flex_kernel_options(query.device),
        return_aux=AuxRequest(lse=True),
    )
    return message[0, 0], aux.lse[0, 0]
