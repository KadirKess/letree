from math import sqrt
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from letree.attention import (
    flex_graph_attention,
    readable_graph_attention,
    same_graph_block_mask,
)
from letree.losses import correction_losses
from letree.model import LetreeModel

# Without CUDA, Flex runs uncompiled on CPU and warns that it builds the scores
pytestmark = pytest.mark.filterwarnings("ignore:flex_attention called without")

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Flex has no backward on CPU"
)


def test_readable_attention_computes_independent_graph_blocks():
    query = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0], [3.0, 4.0]])
    key = torch.tensor([[2.0, 0.0], [1.0, 1.0], [-1.0, 2.0], [5.0, -2.0]])
    hidden = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [7.0, 8.0]])

    message, log_z = readable_graph_attention(
        query, key, hidden, torch.tensor([0, 3, 4])
    )

    # The one-node graph scores 3 * 5 + 4 * -2 = 7 against itself
    first_scores = query[:3] @ key[:3].T / sqrt(2)
    torch.testing.assert_close(
        log_z,
        torch.cat((torch.logsumexp(first_scores, 1), torch.tensor([7 / sqrt(2)]))),
    )
    torch.testing.assert_close(
        message, torch.cat((F.softmax(first_scores, 1) @ hidden[:3], hidden[3:]))
    )


def _packed_inputs(device="cpu", dtype=torch.float64, width=8):
    # Unequal graphs, a one-node graph and one longer than a 128-node tile
    generator = torch.Generator().manual_seed(3)
    graph_ptr = torch.tensor([0, 5, 6, 200, 333], device=device)
    query, key, hidden = (
        torch.randn(333, width, generator=generator, dtype=dtype).to(device)
        for _ in range(3)
    )
    return query, key, hidden, graph_ptr


def test_flex_matches_dense_on_unequal_packed_graphs():
    query, key, hidden, graph_ptr = _packed_inputs()

    actual = flex_graph_attention(query, key, hidden, same_graph_block_mask(graph_ptr))
    expected = readable_graph_attention(query, key, hidden, graph_ptr)

    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


def test_flex_one_node_graph_attends_only_to_itself():
    query, key, hidden, graph_ptr = _packed_inputs()

    message, log_z = flex_graph_attention(
        query, key, hidden, same_graph_block_mask(graph_ptr)
    )

    torch.testing.assert_close(message[5], hidden[5])
    torch.testing.assert_close(log_z[5], query[5] @ key[5] / key.shape[1] ** 0.5)


def test_flex_never_mixes_graphs():
    query, key, hidden, graph_ptr = _packed_inputs()
    mask = same_graph_block_mask(graph_ptr)
    changed = [tensor.clone() for tensor in (query, key, hidden)]
    for tensor in changed:
        tensor[200:] += 50

    original = flex_graph_attention(query, key, hidden, mask)
    after = flex_graph_attention(*changed, mask)

    for a, o in zip(after, original, strict=True):
        torch.testing.assert_close(a[:200], o[:200], rtol=0, atol=0)


def _model_pair(device="cpu", dtype=torch.float64):
    torch.manual_seed(5)
    dense = LetreeModel(8).to(dtype)
    flex = LetreeModel(8, attention="flex").to(dtype)
    flex.load_state_dict(dense.state_dict(), strict=True)
    return dense.to(device), flex.to(device)


def _model_inputs(device="cpu", dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    features = torch.rand(333, 3, generator=generator, dtype=dtype)
    # A chain inside each graph with self loops, enough for the recurrent update
    graph_ptr = torch.tensor([0, 5, 6, 200, 333])
    sources, destinations = [], []
    for start, stop in zip(graph_ptr[:-1].tolist(), graph_ptr[1:].tolist()):
        for node in range(start, stop):
            sources.append(node)
            destinations.append(node)
            if node + 1 < stop:
                sources += [node, node + 1]
                destinations += [node + 1, node]
    edge_index = torch.tensor([sources, destinations])
    return features.to(device), edge_index.to(device), graph_ptr.to(device)


def test_flex_model_matches_dense_model_forward():
    dense, flex = _model_pair()
    inputs = _model_inputs()

    with torch.no_grad():
        expected = dense(*inputs, observed_steps=2, correction_steps=2)
        actual = flex(*inputs, observed_steps=2, correction_steps=2)

    for a, e in zip(
        actual.observed + actual.corrective,
        expected.observed + expected.corrective,
        strict=True,
    ):
        torch.testing.assert_close(a.message, e.message)
        torch.testing.assert_close(a.log_z, e.log_z)
    for a, e in zip(actual.actions, expected.actions, strict=True):
        torch.testing.assert_close(a, e)


@cuda
def test_flex_gradients_match_dense_on_cuda():
    inputs = _packed_inputs("cuda", torch.float32)
    graph_ptr = inputs[3]

    def gradients(attention, *extra):
        query, key, hidden = (x.clone().requires_grad_() for x in inputs[:3])
        message, log_z = attention(query, key, hidden, *extra)
        loss = message.square().sum() + log_z.sum()
        return torch.autograd.grad(loss, (query, key, hidden))

    expected = gradients(readable_graph_attention, graph_ptr)
    actual = gradients(flex_graph_attention, same_graph_block_mask(graph_ptr))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)


@cuda
def test_flex_model_parameter_gradients_match_dense_on_cuda():
    dense, flex = _model_pair("cuda", torch.float32)
    features, edge_index, graph_ptr = _model_inputs("cuda", torch.float32)
    sizes = graph_ptr.diff()
    # Local parent targets that point at the previous node of each graph
    targets = torch.cat([torch.arange(int(n), device="cuda").roll(1) for n in sizes])
    data = SimpleNamespace(
        ptr=graph_ptr,
        batch=torch.repeat_interleave(torch.arange(len(sizes), device="cuda"), sizes),
        num_graphs=len(sizes),
        observed_parents=targets,
        corrective_parents=targets,
        correction_actions=torch.zeros_like(targets),
    )
    for model in (dense, flex):
        output = model(features, edge_index, graph_ptr, 2, 2)
        sum(correction_losses(output, data).values()).backward()

    for (name, a), (_, e) in zip(
        flex.named_parameters(), dense.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, e.grad, rtol=1e-4, atol=1e-5, msg=name)
