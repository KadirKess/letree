"""Recurrent correction model over the flat-zone RAG."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from letree.attention import (
    flex_graph_attention,
    readable_graph_attention,
    round_to_tf32,
    same_graph_block_mask,
)

FEATURE_COUNT = 3


@dataclass
class Step:
    hidden: Tensor
    query: Tensor
    key: Tensor
    log_z: Tensor
    message: Tensor


@dataclass
class Output:
    """Steps of both loops, with node increment and RAG edge contraction logits."""

    observed: list[Step]
    corrective: list[Step]
    increment: list[Tensor]
    contraction: list[Tensor]


def normalized_projection(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width))


class LetreeModel(nn.Module):
    def __init__(
        self,
        width: int,
        correction_recurrent: bool = True,
        aggregations: tuple[str, ...] = ("max", "min"),
        attention: str = "dense",
        tf32_attention: bool = False,
    ) -> None:
        super().__init__()
        if attention not in ("dense", "flex"):
            raise ValueError(f"Unknown attention backend: {attention!r}.")
        self.aggregations = aggregations
        self.attention = attention
        self.tf32_attention = tf32_attention

        gru_input = (2 + len(aggregations)) * width
        self.encoder = nn.Linear(FEATURE_COUNT, width)
        self.observed_gru = nn.GRUCell(gru_input, width)
        # Normalised queries and keys keep the attention scores bounded. Without
        # them the scores grew until training diverged
        self.observed_query = normalized_projection(width)
        self.observed_key = normalized_projection(width)
        self.corrective_gru = (
            nn.GRUCell(gru_input, width) if correction_recurrent else None
        )
        self.corrective_query = normalized_projection(width)
        self.corrective_key = normalized_projection(width)
        self.increment_hidden = nn.Linear(2 * width, width)
        self.increment_output = nn.Linear(width, 1)
        self.contraction_hidden = nn.Linear(2 * width, width)
        self.contraction_output = nn.Linear(width, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        rag_edge_index: Tensor,
        graph_ptr: Tensor,
        observed_steps: int,
        correction_steps: int = 0,
    ) -> Output:
        if correction_steps > 0 and self.corrective_gru is None:
            raise ValueError("Corrective steps need correction_recurrent.")

        embedding = torch.relu(self.encoder(x))
        block_mask = (
            same_graph_block_mask(graph_ptr) if self.attention == "flex" else None
        )

        def attend(hidden, query_layer, key_layer):
            query, key = query_layer(hidden), key_layer(hidden)
            # Under TF32 the kernel, the loss and the metrics must read the same
            # scores, or the loss rewards growing them
            if self.tf32_attention:
                query, key = round_to_tf32(query), round_to_tf32(key)
            if block_mask is not None:
                message, log_z = flex_graph_attention(query, key, hidden, block_mask)
            else:
                message, log_z = readable_graph_attention(query, key, hidden, graph_ptr)
            return Step(hidden, query, key, log_z, message)

        hidden = torch.zeros_like(embedding)
        message = torch.zeros_like(embedding)
        observed = []
        for _ in range(observed_steps):
            hidden = self.update(
                self.observed_gru, hidden, embedding, edge_index, message
            )
            step = attend(hidden, self.observed_query, self.observed_key)
            message = step.message
            observed.append(step)
        if observed_steps == 0:
            hidden = embedding

        # Corrective loop. The first readout comes before any corrective update,
        # and increments read the corrective message, not the observed one.
        source, destination = rag_edge_index
        corrective, increment, contraction = [], [], []
        for index in range(correction_steps + 1):
            if index > 0:
                hidden = self.update(
                    self.corrective_gru,
                    hidden,
                    embedding,
                    edge_index,
                    corrective[-1].message,
                )
            step = attend(hidden, self.corrective_query, self.corrective_key)
            context = torch.relu(
                self.increment_hidden(torch.cat((hidden, step.message), 1))
            )

            corrective.append(step)
            increment.append(self.increment_output(context).squeeze(1))

            # Each RAG edge is scored once. The sum and absolute difference make
            # the score the same whichever end comes first
            first, second = hidden[source], hidden[destination]
            pair = torch.cat((first + second, (first - second).abs()), 1)
            contraction.append(
                self.contraction_output(
                    torch.relu(self.contraction_hidden(pair))
                ).squeeze(1)
            )

        return Output(observed, corrective, increment, contraction)

    def update(
        self,
        gru: nn.GRUCell,
        hidden: Tensor,
        embedding: Tensor,
        edge_index: Tensor,
        message: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        neighbourhood = [
            scatter(
                hidden[source],
                destination,
                dim=0,
                dim_size=hidden.shape[0],
                reduce=reduce,
            )
            for reduce in self.aggregations
        ]
        return gru(torch.cat((message, *neighbourhood, embedding), dim=1), hidden)
