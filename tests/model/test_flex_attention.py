from types import SimpleNamespace

import pytest
import torch

from letree.model import (
    AllNodeParentHead,
    CorrectionModelConfig,
    build_correction_model,
    flex_graph_attention,
    readable_graph_attention,
    same_graph_block_mask,
)
from letree.training.losses import correction_losses

# Without CUDA, Flex runs uncompiled on CPU and warns that it builds the scores
pytestmark = pytest.mark.filterwarnings("ignore:flex_attention called without")

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Flex has no backward on CPU"
)


def _packed_inputs(device="cpu", dtype=torch.float64, width=8):
    # Unequal graphs, a one-node graph and one longer than a 128-node tile
    generator = torch.Generator().manual_seed(3)
    graph_ptr = torch.tensor([0, 5, 6, 200, 333], device=device)
    query, key, hidden_state = (
        torch.randn(333, width, generator=generator, dtype=dtype).to(device)
        for _ in range(3)
    )
    return query, key, hidden_state, graph_ptr


def _assert_same_attention(actual, expected) -> None:
    torch.testing.assert_close(actual.parent_message, expected.parent_message)
    torch.testing.assert_close(actual.log_normalizers, expected.log_normalizers)


def test_flex_matches_dense_on_unequal_packed_graphs() -> None:
    query, key, hidden_state, graph_ptr = _packed_inputs()

    _assert_same_attention(
        flex_graph_attention(
            query, key, hidden_state, same_graph_block_mask(graph_ptr)
        ),
        readable_graph_attention(query, key, hidden_state, graph_ptr),
    )


def test_one_node_graph_attends_only_to_itself() -> None:
    query, key, hidden_state, graph_ptr = _packed_inputs()

    output = flex_graph_attention(
        query, key, hidden_state, same_graph_block_mask(graph_ptr)
    )

    torch.testing.assert_close(output.parent_message[5], hidden_state[5])
    torch.testing.assert_close(
        output.log_normalizers[5], query[5] @ key[5] / key.shape[1] ** 0.5
    )


def test_flex_never_mixes_graphs() -> None:
    query, key, hidden_state, graph_ptr = _packed_inputs()
    mask = same_graph_block_mask(graph_ptr)
    changed = [tensor.clone() for tensor in (query, key, hidden_state)]
    for tensor in changed:
        tensor[200:] += 50

    original = flex_graph_attention(query, key, hidden_state, mask)
    after = flex_graph_attention(*changed, mask)

    torch.testing.assert_close(
        after.parent_message[:200], original.parent_message[:200], rtol=0, atol=0
    )
    torch.testing.assert_close(
        after.log_normalizers[:200], original.log_normalizers[:200], rtol=0, atol=0
    )


def test_head_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="attention backend"):
        AllNodeParentHead(4, "sparse")


def _model_pair(device="cpu", dtype=torch.float64):
    torch.manual_seed(5)
    dense = build_correction_model(CorrectionModelConfig(8, True)).to(dtype)
    flex = build_correction_model(CorrectionModelConfig(8, True), "flex").to(dtype)
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


def test_flex_model_matches_dense_model_forward() -> None:
    dense, flex = _model_pair()
    inputs = _model_inputs()

    with torch.no_grad():
        expected = dense(*inputs, observed_steps=2, correction_steps=2)
        actual = flex(*inputs, observed_steps=2, correction_steps=2)

    for a, e in zip(actual.observed_steps, expected.observed_steps, strict=True):
        torch.testing.assert_close(a.parent_message, e.parent_message)
        torch.testing.assert_close(a.parent_log_normalizers, e.parent_log_normalizers)
    for a, e in zip(actual.correction_steps, expected.correction_steps, strict=True):
        torch.testing.assert_close(
            a.expected_destination_state, e.expected_destination_state
        )
        torch.testing.assert_close(
            a.corrective_log_normalizers, e.corrective_log_normalizers
        )
        torch.testing.assert_close(a.action_logits, e.action_logits)


@cuda
def test_flex_gradients_match_dense_on_cuda() -> None:
    inputs = _packed_inputs("cuda", torch.float32)
    graph_ptr = inputs[3]

    def gradients(attention, *extra):
        query, key, hidden_state = (x.clone().requires_grad_() for x in inputs[:3])
        output = attention(query, key, hidden_state, *extra)
        loss = output.parent_message.square().sum() + output.log_normalizers.sum()
        return torch.autograd.grad(loss, (query, key, hidden_state))

    expected = gradients(readable_graph_attention, graph_ptr)
    actual = gradients(flex_graph_attention, same_graph_block_mask(graph_ptr))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)


@cuda
def test_flex_model_parameter_gradients_match_dense_on_cuda() -> None:
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
        correction_losses(output, data).total().backward()

    for (name, a), (_, e) in zip(
        flex.named_parameters(), dense.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, e.grad, rtol=1e-4, atol=1e-5, msg=name)
