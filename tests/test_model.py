from math import sqrt

import pytest
import torch

from letree.model import LetreeModel


def _inputs():
    # A three-node path and a one-node graph, both with self loops
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]])
    return features, edges, torch.tensor([0, 3, 4])


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


def _readout(query_layer, key_layer, hidden, bounds):
    query, key = _linear(query_layer, hidden), _linear(key_layer, hidden)
    messages, log_z = [], []
    for start, stop in bounds:
        scores = query[start:stop] @ key[start:stop].T / sqrt(hidden.shape[1])
        messages.append(torch.softmax(scores, dim=1) @ hidden[start:stop])
        log_z.append(torch.logsumexp(scores, dim=1))
    return query, key, torch.cat(log_z), torch.cat(messages)


def _reference(model, features, edges, observed_steps, correction_steps):
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

    corrective, actions = [], []
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
            _linear(model.action_hidden, torch.cat((hidden, step[-1]), 1))
        )
        corrective.append(step)
        actions.append(_linear(model.action_output, context))
    return observed, corrective, actions


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
    features, edges, ptr = _inputs()
    features.requires_grad_()

    output = model(features, edges, ptr, observed_steps, correction_steps)
    observed, corrective, actions = _reference(
        model, features, edges, observed_steps, correction_steps
    )

    assert len(output.observed) == observed_steps
    assert len(output.corrective) == len(output.actions) == correction_steps + 1
    for actual, expected in zip(output.observed, observed, strict=True):
        _assert_step(actual, expected)
    for actual, expected in zip(output.corrective, corrective, strict=True):
        _assert_step(actual, expected)
    for actual, expected in zip(output.actions, actions, strict=True):
        torch.testing.assert_close(actual, expected)

    def loss(log_z, message, logits):
        return log_z.square().sum() + message.square().sum() + logits.square().sum()

    actual_loss = loss(
        output.corrective[-1].log_z, output.corrective[-1].message, output.actions[-1]
    )
    expected_loss = loss(corrective[-1][3], corrective[-1][4], actions[-1])
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
        model.observed_query.weight.fill_(1.5)
        model.observed_query.bias.fill_(0.2)
        model.observed_key.weight.fill_(-0.5)
        model.observed_key.bias.fill_(0.4)
    features = torch.tensor([[0.2, 0.3, 0.1], [0.7, 0.1, 0.4]])

    step = model(
        features, torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]]), torch.tensor([0, 2]), 1
    ).observed[0]

    # Only the embedding reaches the GRU, and the previous state is zero
    embedding = torch.relu(features @ torch.tensor([[1.0], [2.0], [-1.0]]) + 0.1)
    reset = torch.sigmoid(0.2 * embedding + 0.1 + 0.2)
    update = torch.sigmoid(-0.3 * embedding - 0.2 + 0.3)
    candidate = torch.tanh(0.5 * embedding + 0.05 - 0.1 * reset)
    hidden = (1.0 - update) * candidate
    scores = (1.5 * hidden + 0.2) @ (-0.5 * hidden + 0.4).T
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
    features, edges, ptr = _inputs()
    permutation = torch.tensor([6, 2, 4, 0, 7, 5, 3, 1])

    original = model(features, edges, ptr, 2, 2)
    reordered = model(features, edges[:, permutation], ptr, 2, 2)

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
    features, edges, ptr = _inputs()
    changed_features = features.clone()
    changed_features[3] += 10

    original = model(features, edges, ptr, observed_steps, 2)
    changed = model(changed_features, edges, ptr, observed_steps, 2)

    for before, after in zip(original.corrective, changed.corrective, strict=True):
        for name in ("hidden", "log_z", "message"):
            torch.testing.assert_close(
                getattr(before, name)[:3], getattr(after, name)[:3], rtol=0, atol=0
            )
    for before, after in zip(original.actions, changed.actions, strict=True):
        torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=0)


def test_one_node_graph_is_its_own_parent():
    torch.manual_seed(12)
    model = LetreeModel(5).double()
    features, edges, ptr = _inputs()

    step = model(features, edges, ptr, 1).observed[0]

    query, key = step.query[3], step.key[3]
    torch.testing.assert_close(step.log_z[3], query @ key / sqrt(5))
    torch.testing.assert_close(step.message[3], step.hidden[3])


def test_zero_observed_steps_start_from_the_embedding_and_skip_observed_layers():
    model = _model()
    features, edges, ptr = _inputs()

    output = model(features, edges, ptr, 0, 1)
    output.actions[-1].square().sum().backward()

    torch.testing.assert_close(
        output.corrective[0].hidden, torch.relu(_linear(model.encoder, features))
    )
    observed_layers = (model.observed_gru, model.observed_query, model.observed_key)
    for layer in observed_layers:
        assert all(parameter.grad is None for parameter in layer.parameters())
    assert model.encoder.weight.grad is not None


def test_later_corrective_steps_read_earlier_messages_but_not_actions():
    torch.manual_seed(47)
    model = LetreeModel(3).double()
    features, edges, ptr = _inputs()

    output = model(features, edges, ptr, 2, 2)
    loss = output.corrective[-1].log_z.square().sum()
    first = output.corrective[0]
    message, query, action = torch.autograd.grad(
        loss, (first.message, first.query, output.actions[0]), allow_unused=True
    )

    assert torch.count_nonzero(message) and torch.count_nonzero(query)
    assert action is None


@pytest.mark.parametrize("loss_name", ["action", "corrective"])
def test_every_used_layer_gets_a_gradient(loss_name):
    torch.manual_seed(29)
    model = LetreeModel(4).double()
    features, edges, ptr = _inputs()

    output = model(features, edges, ptr, 3, 1)
    if loss_name == "action":
        output.actions[-1].square().sum().backward()
    else:
        output.corrective[-1].log_z.square().sum().backward()

    unused = {"action_hidden", "action_output"} if loss_name == "corrective" else set()
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
        "action_hidden",
        "action_output",
    }
    assert "corrective_gru" not in dict(
        LetreeModel(3, correction_recurrent=False).named_children()
    )


def test_invalid_settings_are_rejected():
    features, edges, ptr = _inputs()
    with pytest.raises(ValueError, match="attention backend"):
        LetreeModel(4, attention="sparse")
    with pytest.raises(ValueError, match="correction_recurrent"):
        LetreeModel(4, correction_recurrent=False).double()(features, edges, ptr, 1, 1)
