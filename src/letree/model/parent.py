"""Graph-local parent prediction heads."""

from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from letree.model.attention import (
    ParentAttentionOutput,
    flex_graph_attention,
    readable_graph_attention,
)

ATTENTION_BACKENDS = ("dense", "flex")


class AllNodeParentHead(nn.Module):
    def __init__(self, hidden_width: int, attention: str = "dense") -> None:
        super().__init__()
        if attention not in ATTENTION_BACKENDS:
            raise ValueError(f"Unknown attention backend: {attention!r}.")
        self.attention = attention
        self.query_projection = nn.Linear(hidden_width, hidden_width)
        self.key_projection = nn.Linear(hidden_width, hidden_width)

    def forward(
        self,
        hidden_state: Tensor,
        graph_ptr: Tensor,
        block_mask: BlockMask | None = None,
    ) -> ParentAttentionOutput:
        query = self.query_projection(hidden_state)
        key = self.key_projection(hidden_state)
        if self.attention == "flex":
            return flex_graph_attention(query, key, hidden_state, block_mask)
        return readable_graph_attention(query, key, hidden_state, graph_ptr)
