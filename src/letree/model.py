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
    observed: list[Step]
    corrective: list[Step]
    actions: list[Tensor]


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
        self.observed_query = nn.Linear(width, width)
        self.observed_key = nn.Linear(width, width)
        self.corrective_gru = (
            nn.GRUCell(gru_input, width) if correction_recurrent else None
        )
        self.corrective_query = nn.Linear(width, width)
        self.corrective_key = nn.Linear(width, width)
        self.action_hidden = nn.Linear(2 * width, width)
        self.action_output = nn.Linear(width, 3)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
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
        # and actions read the corrective message, not the observed one.
        corrective, actions = [], []
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
                self.action_hidden(torch.cat((hidden, step.message), 1))
            )
            corrective.append(step)
            actions.append(self.action_output(context))

        return Output(observed, corrective, actions)

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
