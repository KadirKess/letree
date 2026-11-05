from types import SimpleNamespace

import pytest
import torch

from letree.experiments.metrics import (
    batch_image_metrics,
    graph_local_argmax,
    summarize_image_metrics,
)


def test_argmax_returns_local_ids_and_first_index_on_ties() -> None:
    # Graph A has two nodes. Graph B has three, and its first node ties
    query = torch.tensor([[1.0], [1.0], [1.0], [1.0], [-1.0]])
    key = torch.tensor([[0.0], [2.0], [3.0], [3.0], [1.0]])

    predictions, tied = graph_local_argmax(query, key, torch.tensor([0, 2, 5]))

    assert predictions.tolist() == [1, 1, 0, 0, 2]
    assert tied.tolist() == [False, False, True, True, False]


def _metric_batch(values, node_counts, corruption, edge_counts=None):
    names = (
        "observed_parent_accuracy",
        "set_pointer_accuracy",
        "increment_accuracy",
        "joint_accuracy",
        "edge_accuracy",
    )
    batch = {name: torch.tensor(values) for name in names}
    batch["observed_ties"] = torch.zeros(len(values), dtype=torch.long)
    batch["corrective_ties"] = torch.zeros(len(values), dtype=torch.long)
    batch["node_count"] = torch.tensor(node_counts)
    batch["edge_count"] = torch.tensor(edge_counts or node_counts)
    batch["corruption"] = corruption
    return batch


def test_summary_reports_image_mean_and_pooled_over_unequal_batches() -> None:
    # Graph A has 2 nodes and half correct. Graph B, alone in a smaller final
    # batch, has 3 nodes all correct.
    batches = [
        _metric_batch([0.5], [2], ["clean"]),
        _metric_batch([1.0], [3], ["salt_and_pepper"]),
    ]
    per_image = {name: [b[name] for b in batches] for name in batches[0]}

    summary = summarize_image_metrics(per_image)

    assert summary["all/set_pointer_accuracy/image_mean"] == pytest.approx(0.75)
    assert summary["all/set_pointer_accuracy/pooled"] == pytest.approx(0.8)
    assert summary["clean/set_pointer_accuracy/image_mean"] == pytest.approx(0.5)
    assert summary["noisy/set_pointer_accuracy/pooled"] == pytest.approx(1.0)
    assert (summary["all/images"], summary["all/nodes"]) == (2, 5)


def test_clean_only_summary_has_no_noisy_group() -> None:
    per_image = {
        name: [value] for name, value in _metric_batch([1.0], [4], ["clean"]).items()
    }

    summary = summarize_image_metrics(per_image)

    assert "clean/increment_accuracy/image_mean" in summary
    assert {name.split("/")[0] for name in summary} == {"all", "clean"}


def test_summary_reports_each_noise_kind_and_pools_them_as_noisy() -> None:
    # Clean A has 2 nodes and half correct. Salt-and-pepper B has 4 nodes all
    # correct and D, in the next batch, 1 node wrong. Gaussian C has 2 nodes wrong.
    batches = [
        _metric_batch(
            [0.5, 1.0, 0.0],
            [2, 4, 2],
            ["clean", "salt_and_pepper", "additive_gaussian"],
        ),
        _metric_batch([0.0], [1], ["salt_and_pepper"]),
    ]
    per_image = {name: [b[name] for b in batches] for name in batches[0]}

    summary = summarize_image_metrics(per_image)

    accuracy = "set_pointer_accuracy"
    assert summary[f"salt_and_pepper/{accuracy}/image_mean"] == pytest.approx(0.5)
    assert summary[f"salt_and_pepper/{accuracy}/pooled"] == pytest.approx(0.8)
    assert (summary["salt_and_pepper/images"], summary["salt_and_pepper/nodes"]) == (
        2,
        5,
    )
    assert summary[f"additive_gaussian/{accuracy}/image_mean"] == pytest.approx(0.0)
    assert summary["additive_gaussian/images"] == 1
    assert summary[f"noisy/{accuracy}/image_mean"] == pytest.approx(1 / 3)
    assert summary[f"noisy/{accuracy}/pooled"] == pytest.approx(4 / 7)
    assert summary["noisy/images"] == 3
    assert summary[f"clean/{accuracy}/image_mean"] == pytest.approx(0.5)
    # Kinds absent from the split get no entries
    assert {name.split("/")[0] for name in summary} == {
        "all",
        "clean",
        "noisy",
        "salt_and_pepper",
        "additive_gaussian",
    }


def _worked_example(query, key, increment_logits, contraction_logits):
    # The worked example of the zone plan, five nodes on a path whose parent
    # zone is A, made of nodes 0 and 1
    step = SimpleNamespace(query=query, key=key)
    output = SimpleNamespace(
        observed=[step],
        corrective=[step],
        increment=[torch.tensor(increment_logits)],
        contraction=[torch.tensor(contraction_logits)],
    )
    data = SimpleNamespace(
        num_graphs=1,
        ptr=torch.tensor([0, 5]),
        batch=torch.zeros(5, dtype=torch.long),
        observed_parents=torch.zeros(5, dtype=torch.long),
        zone=torch.tensor([0, 0, 2, 2, 4]),
        parent_zone=torch.zeros(5, dtype=torch.long),
        increment=torch.tensor([0, 0, 1, 1, 0]),
        rag_edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
    )
    return batch_image_metrics(output, data)


def test_pointer_is_right_anywhere_in_the_parent_zone_and_joint_needs_both() -> None:
    # Nodes 0, 1 and 2 point at node 1, node 3 at node 0 and node 4 at node 2.
    # Increments are right on nodes 0, 1, 3 and 4
    key = torch.eye(5)
    query = key[[1, 1, 1, 0, 2]]
    metrics = _worked_example(
        query, key, [-1.0, -1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]
    )

    assert metrics["set_pointer_accuracy"].item() == pytest.approx(4 / 5)
    assert metrics["increment_accuracy"].item() == pytest.approx(4 / 5)
    # Node 2 has its increment wrong and node 4 its pointer
    assert metrics["joint_accuracy"].item() == pytest.approx(3 / 5)
    assert metrics["edge_accuracy"].item() == pytest.approx(1.0)


def test_edge_accuracy_reads_the_sign_against_same_zone_targets() -> None:
    query, key = torch.ones(5, 1), torch.ones(5, 1)
    # The plan's false contraction of edge (3, 4)
    metrics = _worked_example(query, key, [0.0] * 5, [1.0, -1.0, 1.0, 2.0])

    assert metrics["edge_accuracy"].item() == pytest.approx(3 / 4)
    assert metrics["edge_count"].tolist() == [4]
    assert metrics["node_count"].tolist() == [5]


def test_edge_accuracy_is_pooled_by_edges_and_skips_images_without_edges() -> None:
    # Image A has 2 nodes, 1 edge and edge accuracy 1. Image B has 4 nodes,
    # 3 edges and accuracy 0. Image C has 1 node and no edge
    per_image = {
        name: [value]
        for name, value in _metric_batch(
            [1.0, 0.0, 0.0], [2, 4, 1], ["clean"] * 3, edge_counts=[1, 3, 0]
        ).items()
    }

    summary = summarize_image_metrics(per_image)

    assert summary["all/edge_accuracy/image_mean"] == pytest.approx(0.5)
    assert summary["all/edge_accuracy/pooled"] == pytest.approx(1 / 4)
    assert summary["all/joint_accuracy/image_mean"] == pytest.approx(1 / 3)
    assert summary["all/joint_accuracy/pooled"] == pytest.approx(2 / 7)
    assert (summary["all/nodes"], summary["all/edges"]) == (7, 4)
