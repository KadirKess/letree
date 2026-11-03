from math import exp, log, sqrt
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from letree.component_tree import ComponentTreeKind
from letree.data import to_graph
from letree.dataset.build import build_sample_family
from letree.generation import SaltAndPepperNoise, SquareImageConfig
from letree.losses import correction_losses, image_mean, parent_cross_entropy
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


@pytest.fixture(scope="module")
def graphs():
    families = [
        build_sample_family(
            3,
            index,
            SMALL_IMAGES,
            ComponentTreeKind.MAX_TREE,
            (SaltAndPepperNoise(0.05),),
            2,
        )
        for index in range(2)
    ]
    samples = [s for f in families for s in (f.clean, *(o.sample for o in f.noisy))]
    graphs = [to_graph(sample) for sample in samples]
    for graph in graphs:
        graph.x = graph.x.double()
    return graphs


def _forward(model, graphs, order):
    batch = Batch.from_data_list([graphs[i] for i in order])
    return model(batch.x, batch.edge_index, batch.ptr, 2, 1), batch


def _scores(query, key, start, stop):
    # Dense scores of one graph, independent of the stored log Z
    return query[start:stop] @ key[start:stop].T / sqrt(query.shape[1])


def test_losses_match_dense_cross_entropy_per_image(graphs) -> None:
    torch.manual_seed(0)
    model = LetreeModel(8).double()
    output, batch = _forward(model, graphs, range(len(graphs)))
    observed, corrective = output.observed[-1], output.corrective[-1]
    assert int(batch.corrective_parents[batch.ptr[1] :].max()) > 0

    expected = {"observed_parent": [], "corrective_parent": [], "action": []}
    bounds = batch.ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        expected["observed_parent"].append(
            F.cross_entropy(
                _scores(observed.query, observed.key, start, stop),
                batch.observed_parents[start:stop],
            )
        )
        expected["corrective_parent"].append(
            F.cross_entropy(
                _scores(corrective.query, corrective.key, start, stop),
                batch.corrective_parents[start:stop],
            )
        )
        expected["action"].append(
            F.cross_entropy(
                output.actions[-1][start:stop],
                batch.correction_actions[start:stop],
            )
        )

    losses = correction_losses(output, batch)
    for name, values in expected.items():
        reference = torch.stack(values).mean()
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
