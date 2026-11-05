import json

import numpy as np
import pytest
from torch_geometric.data import Batch

from letree.data.graphs import ObservationDataset
from letree.evaluation.checkpoint import (
    evaluate_checkpoint,
    load_run_model,
    predict,
    run_evaluation,
)
from letree.evaluation.records import SOURCES


def test_every_sample_gets_one_record_per_source(trained_run):
    root, run = trained_run
    records = evaluate_checkpoint(run, root, "test", batch_size=2)

    dataset = ObservationDataset(root, "test", include_noisy=True)
    expected = {(path.name, position) for path, position in dataset.items}
    for source in SOURCES:
        seen = [
            (record["family"], record["position"])
            for record in records
            if record["source"] == source
        ]
        assert sorted(seen) == sorted(expected)
    assert len(records) == len(SOURCES) * len(dataset)
    # Only the hard vote reads the argmax pointers
    assert all(
        record["ties"] == 0 for record in records if record["source"] != "model_hard"
    )

    oracle = [record for record in records if record["source"] == "oracle"]
    assert all(record["valid"] for record in oracle)
    assert all(record["exact"] for record in oracle if record["refining"])


def test_run_evaluation_writes_records_and_summary(trained_run):
    root, run = trained_run
    summary = run_evaluation(run, root, "test")

    assert (run / "test_best_records.csv").read_text().count("\n") == len(
        SOURCES
    ) * 9 + 1
    stored = json.loads((run / "test_best_summary.json").read_text())
    assert stored == pytest.approx(summary)
    assert summary["oracle/clean/every/exact_recovery"] == 1
    assert summary["observed/clean/every/exact_recovery"] == 1
    assert summary["oracle/all/every/root_claims/sample_mean"] == 1


def test_predictions_are_graph_local_distributions(trained_run):
    root, run = trained_run
    model, settings, device = load_run_model(run)
    dataset = ObservationDataset(root, "test", include_noisy=True)
    graphs = [dataset[index] for index in range(4)]
    predictions = predict(model, settings, Batch.from_data_list(graphs).to(device))

    assert len(predictions) == 4
    for graph, prediction in zip(graphs, predictions):
        node_count = graph.num_nodes
        assert prediction.contraction_logits.shape == (graph.rag_edge_index.shape[1],)
        assert prediction.increment_probabilities.shape == (node_count,)
        assert prediction.pointer_probabilities.shape == (node_count, node_count)
        np.testing.assert_allclose(
            prediction.pointer_probabilities.sum(axis=1), 1, rtol=1e-5
        )
        assert (
            (prediction.increment_probabilities > 0)
            & (prediction.increment_probabilities < 1)
        ).all()
        # The hard pointer is the most probable candidate of its row
        rows = np.arange(node_count)
        np.testing.assert_allclose(
            prediction.pointer_probabilities[rows, prediction.pointers],
            prediction.pointer_probabilities.max(axis=1),
        )
