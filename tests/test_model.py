from math import sqrt

import pytest
import torch

from letree.model import LetreeModel


def _inputs():
    # A three-node path and a one-node graph. Message passing reads both edge
    # directions and self loops, the contraction head each RAG edge once
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]])
    rag_edges = torch.tensor([[0, 1], [1, 2]])
    return features, edges, rag_edges, torch.tensor([0, 3, 4])


def _model(width=2, recurrent=True, aggregations=("max", "min")):
    model = LetreeModel(width, recurrent, aggregations).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.linspace(-0.2, 0.3, parameter.numel()).reshape(parameter.shape)
            )
    return model


def _linear(layer, values):
    return values @ layer.weight.T + layer.bias


def _gru(gru, inputs, hidden):
    input_reset, input_update, input_candidate = (
        inputs @ gru.weight_ih.T + gru.bias_ih
    ).chunk(3, dim=1)
    hidden_reset, hidden_update, hidden_candidate = (
        hidden @ gru.weight_hh.T + gru.bias_hh
    ).chunk(3, dim=1)
    reset = torch.sigmoid(input_reset + hidden_reset)
    update = torch.sigmoid(input_update + hidden_update)
    candidate = torch.tanh(input_candidate + reset * hidden_candidate)
    return (1 - update) * candidate + update * hidden


def _update(gru, aggregations, hidden, embedding, message, edges):
    # Neighbourhood extrema node by node, from the edge list
    pooled = {name: [] for name in aggregations}
    for node in range(hidden.shape[0]):
        neighbours = [int(s) for s, d in edges.T if int(d) == node]
        values = torch.stack([hidden[n] for n in neighbours])
        if "max" in pooled:
            pooled["max"].append(values.max(dim=0).values)
        if "min" in pooled:
            pooled["min"].append(values.min(dim=0).values)
    inputs = torch.cat(
        (message, *(torch.stack(pooled[name]) for name in aggregations), embedding), 1
    )
    return _gru(gru, inputs, hidden)


def _normalized_projection(layers, values):
    linear, norm = layers
    projected = _linear(linear, values)
    centred = projected - projected.mean(dim=1, keepdim=True)
    variance = centred.square().mean(dim=1, keepdim=True)
    return centred / torch.sqrt(variance + norm.eps) * norm.weight + norm.bias


def _readout(query_layers, key_layers, hidden, bounds):
    query = _normalized_projection(query_layers, hidden)
    key = _normalized_projection(key_layers, hidden)
    messages, log_z = [], []
    for start, stop in bounds:
        scores = query[start:stop] @ key[start:stop].T / sqrt(hidden.shape[1])
        messages.append(torch.softmax(scores, dim=1) @ hidden[start:stop])
        log_z.append(torch.logsumexp(scores, dim=1))
    return query, key, torch.cat(log_z), torch.cat(messages)


def _reference(model, features, edges, rag_edges, observed_steps, correction_steps):
    """Plain-loop forward pass written from the model equations."""
    bounds = [(0, 3), (3, 4)]
    embedding = torch.relu(_linear(model.encoder, features))
    hidden = torch.zeros_like(embedding)
    message = torch.zeros_like(embedding)
    observed = []
    for _ in range(observed_steps):
        hidden = _update(
            model.observed_gru, model.aggregations, hidden, embedding, message, edges
        )
        readout = _readout(model.observed_query, model.observed_key, hidden, bounds)
        message = readout[-1]
        observed.append((hidden, *readout))
    if observed_steps == 0:
        hidden = embedding

    corrective, increments, contractions = [], [], []
    for index in range(correction_steps + 1):
        if index > 0:
            hidden = _update(
                model.corrective_gru,
                model.aggregations,
                hidden,
                embedding,
                corrective[-1][-1],
                edges,
            )
        readout = _readout(model.corrective_query, model.corrective_key, hidden, bounds)
        step = (hidden, *readout)
        context = torch.relu(
            _linear(model.increment_hidden, torch.cat((hidden, step[-1]), 1))
        )
        corrective.append(step)
        increments.append(_linear(model.increment_output, context)[:, 0])
        # One row per RAG edge from the sum and absolute difference of its ends
        pairs = torch.stack(
            [
                torch.cat((hidden[u] + hidden[v], (hidden[u] - hidden[v]).abs()))
                for u, v in rag_edges.T.tolist()
            ]
        )
        contraction_context = torch.relu(_linear(model.contraction_hidden, pairs))
        contractions.append(
            _linear(model.contraction_output, contraction_context)[:, 0]
        )
    return observed, corrective, increments, contractions


def _assert_step(actual, expected):
    for name, value in zip(("hidden", "query", "key", "log_z", "message"), expected):
        torch.testing.assert_close(getattr(actual, name), value, msg=name)


@pytest.mark.parametrize("aggregations", [("max", "min"), ("max",), ("min",), ()])
@pytest.mark.parametrize(
    ("observed_steps", "correction_steps"), [(0, 0), (0, 2), (3, 0), (2, 2)]
)
def test_forward_and_gradients_match_the_model_equations(
    aggregations, observed_steps, correction_steps
):
    model = _model(aggregations=aggregations)
    features, edges, rag_edges, ptr = _inputs()
    features.requires_grad_()

    output = model(features, edges, rag_edges, ptr, observed_steps, correction_steps)
    observed, corrective, increments, contractions = _reference(
        model, features, edges, rag_edges, observed_steps, correction_steps
    )

    assert len(output.observed) == observed_steps
    assert len(output.corrective) == correction_steps + 1
    assert len(output.increment) == len(output.contraction) == correction_steps + 1
    for actual, expected in zip(output.observed, observed, strict=True):
        _assert_step(actual, expected)
    for actual, expected in zip(output.corrective, corrective, strict=True):
        _assert_step(actual, expected)
    for actual, expected in zip(output.increment, increments, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(output.contraction, contractions, strict=True):
        torch.testing.assert_close(actual, expected)

    def loss(log_z, message, increment, contraction):
        return (
            log_z.square().sum()
            + message.square().sum()
            + increment.square().sum()
            + contraction.square().sum()
        )

    last = output.corrective[-1]
    actual_loss = loss(
        last.log_z, last.message, output.increment[-1], output.contraction[-1]
    )
    expected_loss = loss(
        corrective[-1][3], corrective[-1][4], increments[-1], contractions[-1]
    )
    inputs = (features, *model.parameters())
    actual_gradients = torch.autograd.grad(
        actual_loss, inputs, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(expected_loss, inputs, allow_unused=True)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)


def test_one_step_matches_a_hand_calculation():
    model = LetreeModel(1, correction_recurrent=False)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
        model.encoder.bias.fill_(0.1)
        model.observed_gru.weight_ih.copy_(
            torch.tensor(
                [[0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, -0.3], [0.0, 0.0, 0.0, 0.5]]
            )
        )
        model.observed_gru.weight_hh.zero_()
        model.observed_gru.bias_ih.copy_(torch.tensor([0.1, -0.2, 0.05]))
        model.observed_gru.bias_hh.copy_(torch.tensor([0.2, 0.3, -0.1]))
        model.observed_query[0].weight.fill_(1.5)
        model.observed_query[0].bias.fill_(0.2)
        model.observed_query[1].bias.fill_(0.3)
        model.observed_key[0].weight.fill_(-0.5)
        model.observed_key[0].bias.fill_(0.4)
        model.observed_key[1].bias.fill_(-2.0)
    features = torch.tensor([[0.2, 0.3, 0.1], [0.7, 0.1, 0.4]])

    step = model(
        features,
        torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]]),
        torch.tensor([[0], [1]]),
        torch.tensor([0, 2]),
        1,
    ).observed[0]

    # Only the embedding reaches the GRU, and the previous state is zero
    embedding = torch.relu(features @ torch.tensor([[1.0], [2.0], [-1.0]]) + 0.1)
    reset = torch.sigmoid(0.2 * embedding + 0.1 + 0.2)
    update = torch.sigmoid(-0.3 * embedding - 0.2 + 0.3)
    candidate = torch.tanh(0.5 * embedding + 0.05 - 0.1 * reset)
    hidden = (1.0 - update) * candidate
    # With one feature the normalisation leaves only its bias, so every score
    # is 0.3 times -2
    scores = torch.full((2, 2), -0.6)
    torch.testing.assert_close(step.hidden, hidden)
    torch.testing.assert_close(step.log_z, torch.logsumexp(scores, dim=1))
    torch.testing.assert_close(step.message, torch.softmax(scores, dim=1) @ hidden)


@pytest.mark.parametrize(
    ("edge_index", "expected_maximum", "expected_minimum"),
    [
        (
            torch.tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]),
            [[3.0, 4.0], [5.0, 4.0], [5.0, 4.0]],
            [[1.0, 2.0], [1.0, 0.0], [3.0, 0.0]],
        ),
        (
            torch.tensor(
                [[0, 1, 0, 2, 0, 3, 0, 1, 2, 3], [1, 0, 2, 0, 3, 0, 0, 1, 2, 3]]
            ),
            [[7.0, 8.0], [3.0, 4.0], [5.0, 2.0], [7.0, 8.0]],
            [[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 2.0]],
        ),
        (torch.tensor([[0], [0]]), [[1.0, 2.0]], [[1.0, 2.0]]),
    ],
    ids=["path", "star", "one_node"],
)
def test_aggregations_are_neighbourhood_extrema_in_input_order(
    edge_index, expected_maximum, expected_minimum
):
    node_count = int(edge_index.max()) + 1
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [7.0, 8.0]])[:node_count]
    message, embedding = torch.full_like(hidden, -1.0), torch.full_like(hidden, -2.0)
    captured = []

    def capture(inputs, previous):
        captured.append(inputs)
        return previous

    for aggregations in (("max", "min"), ("min",)):
        LetreeModel(2, aggregations=aggregations).update(
            capture, hidden, embedding, edge_index, message
        )

    torch.testing.assert_close(
        captured[0],
        torch.cat(
            (
                message,
                torch.tensor(expected_maximum),
                torch.tensor(expected_minimum),
                embedding,
            ),
            dim=1,
        ),
    )
    torch.testing.assert_close(
        captured[1], torch.cat((message, torch.tensor(expected_minimum), embedding), 1)
    )


def test_edge_reordering_does_not_change_outputs():
    torch.manual_seed(11)
    model = LetreeModel(4).double()
    features, edges, rag_edges, ptr = _inputs()
    permutation = torch.tensor([6, 2, 4, 0, 7, 5, 3, 1])

    original = model(features, edges, rag_edges, ptr, 2, 2)
    reordered = model(features, edges[:, permutation], rag_edges, ptr, 2, 2)

    for a, b in zip(
        original.observed + original.corrective,
        reordered.observed + reordered.corrective,
        strict=True,
    ):
        _assert_step(a, (b.hidden, b.query, b.key, b.log_z, b.message))


@pytest.mark.parametrize("observed_steps", [0, 2])
def test_graphs_in_a_batch_never_mix(observed_steps):
    torch.manual_seed(59)
    model = LetreeModel(3).double()
    features, edges, rag_edges, ptr = _inputs()
    changed_features = features.clone()
    changed_features[3] += 10

    original = model(features, edges, rag_edges, ptr, observed_steps, 2)
    changed = model(changed_features, edges, rag_edges, ptr, observed_steps, 2)

    for before, after in zip(original.corrective, changed.corrective, strict=True):
        for name in ("hidden", "log_z", "message"):
            torch.testing.assert_close(
                getattr(before, name)[:3], getattr(after, name)[:3], rtol=0, atol=0
            )
    for before, after in zip(original.increment, changed.increment, strict=True):
        torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=0)
    # Both RAG edges lie in the first graph
    for before, after in zip(original.contraction, changed.contraction, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_one_node_graph_is_its_own_parent():
    torch.manual_seed(12)
    model = LetreeModel(5).double()
    step = model(*_inputs(), 1).observed[0]

    query, key = step.query[3], step.key[3]
    torch.testing.assert_close(step.log_z[3], query @ key / sqrt(5))
    torch.testing.assert_close(step.message[3], step.hidden[3])


def test_zero_observed_steps_start_from_the_embedding_and_skip_observed_layers():
    model = _model()
    features, edges, rag_edges, ptr = _inputs()

    output = model(features, edges, rag_edges, ptr, 0, 1)
    output.increment[-1].square().sum().backward()

    torch.testing.assert_close(
        output.corrective[0].hidden, torch.relu(_linear(model.encoder, features))
    )
    observed_layers = (model.observed_gru, model.observed_query, model.observed_key)
    for layer in observed_layers:
        assert all(parameter.grad is None for parameter in layer.parameters())
    assert model.encoder.weight.grad is not None


def test_later_corrective_steps_read_earlier_messages_but_not_head_outputs():
    torch.manual_seed(47)
    model = LetreeModel(3).double()

    output = model(*_inputs(), 2, 2)
    loss = output.corrective[-1].log_z.square().sum()
    first = output.corrective[0]
    message, query, increment, contraction = torch.autograd.grad(
        loss,
        (first.message, first.query, output.increment[0], output.contraction[0]),
        allow_unused=True,
    )

    assert torch.count_nonzero(message) and torch.count_nonzero(query)
    assert increment is None and contraction is None


HEADS = {
    "increment": {"increment_hidden", "increment_output"},
    "contraction": {"contraction_hidden", "contraction_output"},
}


@pytest.mark.parametrize("loss_name", ["increment", "contraction", "corrective"])
def test_every_used_layer_gets_a_gradient(loss_name):
    torch.manual_seed(29)
    model = LetreeModel(4).double()

    output = model(*_inputs(), 3, 1)
    if loss_name == "corrective":
        output.corrective[-1].log_z.square().sum().backward()
    else:
        getattr(output, loss_name)[-1].square().sum().backward()

    unused = set().union(
        *(layers for name, layers in HEADS.items() if name != loss_name)
    )
    for name, parameter in model.named_parameters():
        if name.split(".")[0] in unused:
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_observed_and_corrective_layers_are_separate():
    model = LetreeModel(3)
    names = {name.split(".")[0] for name, _ in model.named_parameters()}

    assert names == {
        "encoder",
        "observed_gru",
        "observed_query",
        "observed_key",
        "corrective_gru",
        "corrective_query",
        "corrective_key",
        "increment_hidden",
        "increment_output",
        "contraction_hidden",
        "contraction_output",
    }
    assert "corrective_gru" not in dict(
        LetreeModel(3, correction_recurrent=False).named_children()
    )


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="attention backend"):
        LetreeModel(4, attention="sparse")
    with pytest.raises(ValueError, match="correction_recurrent"):
        LetreeModel(4, correction_recurrent=False).double()(*_inputs(), 1, 1)


def test_tf32_attention_rounds_every_query_and_key():
    torch.manual_seed(0)
    model = LetreeModel(16, tf32_attention=True)
    features, edges, rag_edges, graph_ptr = _inputs()

    output = model(features.float(), edges, rag_edges, graph_ptr, 2, 1)

    for step in output.observed + output.corrective:
        for values in (step.query, step.key):
            assert (values.view(torch.int32) & 0x1FFF).eq(0).all()


def test_query_and_key_lengths_do_not_grow_with_their_weights():
    torch.manual_seed(0)
    model = LetreeModel(16)
    features, edges, rag_edges, graph_ptr = _inputs()
    with torch.no_grad():
        for name in ("observed_query", "observed_key"):
            getattr(model, name)[0].weight.mul_(1_000)

    step = model(features.float(), edges, rag_edges, graph_ptr, 1).observed[0]

    # Untrained normalisation gains are one and biases zero, so every length
    # is the square root of the width
    for values in (step.query, step.key):
        torch.testing.assert_close(
            values.norm(dim=1), torch.full((4,), 4.0), rtol=1e-3, atol=0
        )


def test_contraction_logits_ignore_the_order_of_edge_ends():
    torch.manual_seed(31)
    model = LetreeModel(4).double()
    features, edges, rag_edges, ptr = _inputs()

    original = model(features, edges, rag_edges, ptr, 2, 1)
    flipped = model(features, edges, rag_edges.flip(0), ptr, 2, 1)

    torch.testing.assert_close(
        original.contraction[-1], flipped.contraction[-1], rtol=0, atol=0
    )
