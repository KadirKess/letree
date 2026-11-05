from math import sqrt
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask

from letree.attention import (
    flex_graph_attention,
    flex_target_set_log_z,
    readable_graph_attention,
    readable_target_set_log_z,
    round_to_tf32,
    same_graph_block_mask,
)
from letree.experiments.losses import correction_losses
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


# The CUDA Flex kernel needs at least 16 features per head
WIDTH = 16


def _packed_inputs(device="cpu", dtype=torch.float64, width=WIDTH):
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
    dense = LetreeModel(WIDTH).to(dtype)
    flex = LetreeModel(WIDTH, attention="flex").to(dtype)
    flex.load_state_dict(dense.state_dict(), strict=True)
    return dense.to(device), flex.to(device)


def _model_inputs(device="cpu", dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    features = torch.rand(333, 3, generator=generator, dtype=dtype)
    # A chain inside each graph with self loops, enough for the recurrent update
    graph_ptr = torch.tensor([0, 5, 6, 200, 333])
    sources, destinations, rag_edges = [], [], []
    for start, stop in zip(graph_ptr[:-1].tolist(), graph_ptr[1:].tolist()):
        for node in range(start, stop):
            sources.append(node)
            destinations.append(node)
            if node + 1 < stop:
                sources += [node, node + 1]
                destinations += [node + 1, node]
                rag_edges.append((node, node + 1))
    edge_index = torch.tensor([sources, destinations])
    rag_edge_index = torch.tensor(rag_edges).T
    return (
        features.to(device),
        edge_index.to(device),
        rag_edge_index.to(device),
        graph_ptr.to(device),
    )


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
    for a, e in zip(actual.increment, expected.increment, strict=True):
        torch.testing.assert_close(a, e)
    for a, e in zip(actual.contraction, expected.contraction, strict=True):
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
def test_block_mask_of_a_large_batch_never_needs_n_by_n_memory():
    # 16 graphs and 46,000 nodes, the size of the largest mixed 64 by 64 batches.
    # One byte per node pair would already be about 2 GiB.
    graph_ptr = torch.arange(0, 46_001, 2_875, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    same_graph_block_mask(graph_ptr)

    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < 2**28


@cuda
def test_compiled_block_mask_matches_the_eager_one():
    graph_ptr = torch.tensor([0, 1, 300, 301, 1_000, 2_500], device="cuda")
    graph_ids = torch.repeat_interleave(
        torch.arange(5, device="cuda"), graph_ptr.diff()
    )

    def same_graph(batch, head, query_index, key_index):
        return graph_ids[query_index] == graph_ids[key_index]

    expected = create_block_mask(same_graph, None, None, 2_500, 2_500, device="cuda")
    actual = same_graph_block_mask(graph_ptr)

    torch.testing.assert_close(actual.to_dense(), expected.to_dense())


@cuda
def test_flex_model_parameter_gradients_match_dense_on_cuda():
    dense, flex = _model_pair("cuda", torch.float32)
    features, edge_index, rag_edge_index, graph_ptr = _model_inputs(
        "cuda", torch.float32
    )
    sizes = graph_ptr.diff()
    # Local parent targets that point at the previous node of each graph
    targets = torch.cat([torch.arange(int(n), device="cuda").roll(1) for n in sizes])
    zone, parent_zone = _zones(graph_ptr)
    data = SimpleNamespace(
        ptr=graph_ptr,
        batch=torch.repeat_interleave(torch.arange(len(sizes), device="cuda"), sizes),
        num_graphs=len(sizes),
        rag_edge_index=rag_edge_index,
        observed_parents=targets,
        zone=zone,
        parent_zone=parent_zone,
        increment=(zone != parent_zone).long(),
    )
    for model, attention in ((dense, "dense"), (flex, "flex")):
        output = model(features, edge_index, rag_edge_index, graph_ptr, 2, 2)
        sum(correction_losses(output, data, attention).values()).backward()

    for (name, a), (_, e) in zip(
        flex.named_parameters(), dense.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, e.grad, rtol=1e-4, atol=1e-5, msg=name)


def _truncate_to_tf32(values):
    # What the GPU does to a float32 matmul input under TF32
    return (values.view(torch.int32) & ~0x1FFF).view(torch.float32)


def _large_values(count=1_000, width=128, scale=100.0):
    generator = torch.Generator().manual_seed(4)
    return torch.randn(count, width, generator=generator) * scale


def test_tf32_rounding_is_within_half_a_tf32_step():
    values = _large_values()
    rounded = round_to_tf32(values)

    assert (rounded.view(torch.int32) & 0x1FFF).eq(0).all()
    # TF32 keeps 10 explicit mantissa bits, so a step is 2 ** -10 of the leading power
    exponent = torch.frexp(values.double()).exponent
    half_step = torch.ldexp(torch.ones_like(values.double()), exponent - 12)
    assert ((rounded.double() - values.double()).abs() <= half_step).all()


def test_tf32_rounding_survives_the_gpu_truncation():
    rounded = round_to_tf32(_large_values())
    assert torch.equal(_truncate_to_tf32(rounded), rounded)


def test_products_of_tf32_values_are_exact_in_float32():
    first, second = round_to_tf32(_large_values()).chunk(2)
    assert torch.equal((first * second).double(), first.double() * second.double())


def test_tf32_rounding_passes_the_gradient_through():
    values = _large_values(count=10).requires_grad_()
    weights = torch.randn(10, 128)
    (round_to_tf32(values) * weights).sum().backward()
    assert torch.equal(values.grad, weights)


@cuda
def test_tf32_flex_log_z_matches_float64_on_large_scores():
    torch.set_float32_matmul_precision("high")
    try:
        graph_ptr = torch.tensor([0, 300, 301, 1_000], device="cuda")
        query, key, hidden = (
            round_to_tf32(_large_values(1_000).cuda()) for _ in range(3)
        )
        _, log_z = flex_graph_attention(
            query, key, hidden, same_graph_block_mask(graph_ptr)
        )
        _, expected = readable_graph_attention(
            query.double(), key.double(), hidden.double(), graph_ptr
        )
    finally:
        torch.set_float32_matmul_precision("highest")

    # Scores reach about 40,000. Truncated inputs would lower log Z by about
    # 0.1 percent of that, while rounded ones only differ in summation order.
    assert (expected.abs().max() > 10_000).item()
    torch.testing.assert_close(log_z.double(), expected, rtol=0, atol=0.5)


def _zones(graph_ptr):
    # Local zones of three consecutive nodes, each pointing at the zone with
    # half its id, so ids collide across graphs and every target set exists
    zone = torch.cat(
        [torch.arange(int(n), device=graph_ptr.device) // 3 for n in graph_ptr.diff()]
    )
    return zone, zone // 2


def _dense_target_set_log_z(query, key, zone, parent_zone, graph_ptr):
    # Log of the softmax mass on the target set plus the full log Z, graph by graph
    values = []
    bounds = graph_ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        for i in range(start, stop):
            scores = key[start:stop] @ query[i] / sqrt(query.shape[1])
            members = [
                j for j in range(stop - start) if zone[start + j] == parent_zone[i]
            ]
            mass = F.softmax(scores, 0)[members].sum()
            values.append(torch.log(mass) + torch.logsumexp(scores, 0))
    return torch.stack(values)


def test_target_set_log_z_matches_a_dense_computation():
    query, key, _, graph_ptr = _packed_inputs()
    zone, parent_zone = _zones(graph_ptr)

    expected = _dense_target_set_log_z(query, key, zone, parent_zone, graph_ptr)

    torch.testing.assert_close(
        readable_target_set_log_z(query, key, zone, parent_zone, graph_ptr), expected
    )
    torch.testing.assert_close(
        flex_target_set_log_z(query, key, zone, parent_zone, graph_ptr), expected
    )


@pytest.mark.parametrize(
    "target_set_log_z", [readable_target_set_log_z, flex_target_set_log_z]
)
def test_target_set_of_the_whole_graph_gives_the_full_log_z(target_set_log_z):
    query, key, hidden, graph_ptr = _packed_inputs()
    # One zone per graph, which is its own parent zone
    zone = torch.zeros(333, dtype=torch.long)

    actual = target_set_log_z(query, key, zone, zone, graph_ptr)

    _, log_z = readable_graph_attention(query, key, hidden, graph_ptr)
    torch.testing.assert_close(actual, log_z)


@pytest.mark.parametrize(
    "target_set_log_z", [readable_target_set_log_z, flex_target_set_log_z]
)
def test_target_set_never_reaches_into_another_graph(target_set_log_z):
    query, key, _, graph_ptr = _packed_inputs()
    zone, parent_zone = _zones(graph_ptr)
    changed_query, changed_key = query.clone(), key.clone()
    changed_query[200:] += 50
    changed_key[200:] += 50

    original = target_set_log_z(query, key, zone, parent_zone, graph_ptr)
    after = target_set_log_z(changed_query, changed_key, zone, parent_zone, graph_ptr)

    torch.testing.assert_close(after[:200], original[:200], rtol=0, atol=0)


@cuda
def test_flex_target_set_gradients_match_readable_on_cuda():
    inputs = _packed_inputs("cuda", torch.float32)
    graph_ptr = inputs[3]
    zone, parent_zone = _zones(graph_ptr)

    def gradients(target_set_log_z):
        query, key = (x.clone().requires_grad_() for x in inputs[:2])
        values = target_set_log_z(query, key, zone, parent_zone, graph_ptr)
        return torch.autograd.grad(values.sum(), (query, key))

    expected = gradients(readable_target_set_log_z)
    actual = gradients(flex_target_set_log_z)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)
