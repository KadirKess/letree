"""Recurrent RGNN orchestration and per-step outputs."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from letree.model.attention import same_graph_block_mask


@dataclass(frozen=True)
class ObservedStep:
    hidden_state: Tensor
    parent_message: Tensor
    parent_query: Tensor
    parent_key: Tensor
    parent_log_normalizers: Tensor


class RGNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        recurrent_processor: nn.Module,
        parent_head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent_processor = recurrent_processor
        self.parent_head = parent_head

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        graph_ptr: Tensor,
        steps: int,
    ) -> tuple[ObservedStep, ...]:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer.")
        if steps <= 0:
            raise ValueError("steps must be positive.")

        static_embedding = self.encoder(node_features)
        block_mask = (
            same_graph_block_mask(graph_ptr)
            if self.parent_head.attention == "flex"
            else None
        )
        return self._run_steps(
            static_embedding, edge_index, graph_ptr, block_mask, steps
        )

    def _run_steps(
        self,
        static_embedding: Tensor,
        edge_index: Tensor,
        graph_ptr: Tensor,
        block_mask: BlockMask | None,
        steps: int,
    ) -> tuple[ObservedStep, ...]:
        # Start without a hidden state or an initial parent guess
        hidden_state = torch.zeros_like(static_embedding)
        parent_message = torch.zeros_like(static_embedding)
        outputs = []

        for _ in range(steps):
            hidden_state = self.recurrent_processor(
                hidden_state,
                static_embedding,
                edge_index,
                parent_message,
            )
            parent_output = self.parent_head(hidden_state, graph_ptr, block_mask)

            parent_message = parent_output.parent_message
            outputs.append(
                ObservedStep(
                    hidden_state=hidden_state,
                    parent_message=parent_message,
                    parent_query=parent_output.query,
                    parent_key=parent_output.key,
                    parent_log_normalizers=parent_output.log_normalizers,
                )
            )

        return tuple(outputs)
