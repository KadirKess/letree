from math import exp, log, sqrt
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from letree.data.graphs import to_graph
from letree.experiments.losses import (
    contraction_targets,
    correction_losses,
    image_mean,
    parent_cross_entropy,
    set_pointer_cross_entropy,
)
from letree.generation import SaltAndPepperNoise, SquareImageConfig
from letree.model import LetreeModel, Step

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)


def test_parent_cross_entropy_matches_hand_calculation() -> None:
    # One node whose candidate scores are [2, 0, 1] with width one, target 2
    query = torch.tensor([[1.0], [0.0], [0.0]])
    key = torch.tensor([[2.0], [0.0], [1.0]])
    log_normalizer = torch.tensor([log(exp(2) + exp(0) + exp(1)), 0.0, 0.0])

    step = Step(None, query, key, log_normalizer, None)
    data = SimpleNamespace(ptr=torch.tensor([0, 3]), batch=torch.zeros(3, dtype=int))

    losses = parent_cross_entropy(step, torch.tensor([2, 0, 0]), data)

    assert losses[0].item() == pytest.approx(2.4076 - 1.0, abs=1e-4)


def test_image_mean_weights_images_equally() -> None:
    values = torch.tensor([0.2, 0.4, 1.0, 0.0, 0.5])
    batch = torch.tensor([0, 0, 1, 1, 1])

    data = SimpleNamespace(batch=batch, num_graphs=2)

    assert image_mean(values, data).item() == pytest.approx(0.4)
    assert values.mean().item() == pytest.approx(0.42)


def test_image_mean_skips_images_without_values() -> None:
    # Three edges, all in images 0 and 2, so image 1 has no edge to average
    values = torch.tensor([1.0, 0.0, 0.5])
    data = SimpleNamespace(batch=None, num_graphs=3)

    mean = image_mean(values, data, torch.tensor([0, 0, 2]))

    assert mean.item() == pytest.approx((0.5 + 0.5) / 2)


def _worked_example():
    # The worked example of the zone plan. Clean zones A, B and C are 0, 2 and
    # 4, the observed nodes o0 to o4 lie on a path, and every parent zone is A
    zone = torch.tensor([0, 0, 2, 2, 4])
    return SimpleNamespace(
        ptr=torch.tensor([0, 5]),
        batch=torch.zeros(5, dtype=torch.long),
        num_graphs=1,
        zone=zone,
        parent_zone=torch.zeros(5, dtype=torch.long),
        increment=torch.tensor([0, 0, 1, 1, 0]),
        rag_edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
    )


def test_contraction_targets_of_the_worked_example() -> None:
    assert contraction_targets(_worked_example()).tolist() == [1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("attention", ["dense", "flex"])
def test_set_pointer_loss_is_minus_log_mass_on_the_parent_zone(attention) -> None:
    # Width one, so the score of i on j is query[i] times key[j]. Node 0 scores
    # [1, 2, 0, 0, 3] and puts the mass of its first two entries on zone A
    query = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])
    key = torch.tensor([[1.0], [2.0], [0.0], [0.0], [3.0]])
    log_z = torch.logsumexp(query @ key.T, 1)
    step = Step(None, query, key, log_z, None)

    losses = set_pointer_cross_entropy(step, _worked_example(), attention)

    total = exp(1) + exp(2) + 2 + exp(3)
    assert losses[0].item() == pytest.approx(-log((exp(1) + exp(2)) / total))
    # Nodes 1 to 4 score zero everywhere, so two of five candidates are right
    for value in losses[1:].tolist():
        assert value == pytest.approx(-log(2 / 5))


@pytest.fixture(scope="module")
def graphs(square_family):
    families = [
        square_family(3, index, (SaltAndPepperNoise(0.05),), 2) for index in range(2)
    ]
    graphs = [to_graph(o.tree_image, o.targets) for f in families for o in f]
    for graph in graphs:
        graph.x = graph.x.double()
    return graphs


def _forward(model, graphs, order):
    batch = Batch.from_data_list([graphs[i] for i in order])
    output = model(batch.x, batch.edge_index, batch.rag_edge_index, batch.ptr, 2, 1)
    return output, batch


def _scores(query, key, start, stop):
    # Dense scores of one graph, independent of the stored log Z
    return query[start:stop] @ key[start:stop].T / sqrt(query.shape[1])


def _dense_losses(output, batch):
    """Each loss image by image from dense scores and plain loops."""
    observed, corrective = output.observed[-1], output.corrective[-1]
    expected = {name: [] for name in ("observed_parent", "set_pointer", "contraction")}
    expected["increment"] = []
    source, destination = batch.rag_edge_index
    bounds = batch.ptr.tolist()
    for image, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        expected["observed_parent"].append(
            F.cross_entropy(
                _scores(observed.query, observed.key, start, stop),
                batch.observed_parents[start:stop],
            )
        )
        probabilities = F.softmax(
            _scores(corrective.query, corrective.key, start, stop), 1
        )
        zone, parent_zone = batch.zone[start:stop], batch.parent_zone[start:stop]
        in_target_set = zone[None, :] == parent_zone[:, None]
        expected["set_pointer"].append(
            -torch.log((probabilities * in_target_set).sum(1)).mean()
        )
        edges = (batch.batch[source] == image).nonzero()[:, 0]
        expected["contraction"].append(
            F.binary_cross_entropy_with_logits(
                output.contraction[-1][edges],
                (batch.zone[source[edges]] == batch.zone[destination[edges]]).double(),
            )
        )
        expected["increment"].append(
            F.binary_cross_entropy_with_logits(
                output.increment[-1][start:stop],
                batch.increment[start:stop].double(),
            )
        )
    return {name: torch.stack(values).mean() for name, values in expected.items()}


def test_losses_match_dense_computations_per_image(graphs) -> None:
    torch.manual_seed(0)
    model = LetreeModel(8).double()
    output, batch = _forward(model, graphs, range(len(graphs)))
    # Noisy graphs have contracted edges and zones of several nodes
    assert contraction_targets(batch).sum() > 0
    assert batch.increment.sum() > 0

    losses = correction_losses(output, batch)
    for name, reference in _dense_losses(output, batch).items():
        actual = losses[name]
        assert actual.item() == pytest.approx(reference.item(), rel=1e-12)
        actual_gradients = torch.autograd.grad(
            actual, model.parameters(), retain_graph=True, allow_unused=True
        )
        reference_gradients = torch.autograd.grad(
            reference, model.parameters(), retain_graph=True, allow_unused=True
        )
        for a, b in zip(actual_gradients, reference_gradients, strict=True):
            if a is not None:
                torch.testing.assert_close(a, b)


def test_flex_losses_match_dense_losses(graphs) -> None:
    torch.manual_seed(0)
    model = LetreeModel(8).double()
    with torch.no_grad():
        output, batch = _forward(model, graphs, range(len(graphs)))
        dense = correction_losses(output, batch, "dense")
        flex = correction_losses(output, batch, "flex")

    for name in dense:
        torch.testing.assert_close(flex[name], dense[name])


def test_losses_do_not_depend_on_graph_order(graphs) -> None:
    torch.manual_seed(0)
    model = LetreeModel(8).double()

    forward = sum(
        correction_losses(*_forward(model, graphs, [0, 1, 2, 3, 4, 5])).values()
    )
    shuffled = sum(
        correction_losses(*_forward(model, graphs, [5, 3, 1, 0, 2, 4])).values()
    )

    assert forward.item() == pytest.approx(shuffled.item(), rel=1e-12)
