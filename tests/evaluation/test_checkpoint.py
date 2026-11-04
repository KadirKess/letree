import json

import pytest

from letree.data.graphs import ObservationDataset
from letree.evaluation.checkpoint import evaluate_checkpoint, run_evaluation
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
    assert len(records) == 3 * len(dataset)

    oracle = [record for record in records if record["source"] == "oracle"]
    assert all(record["valid"] for record in oracle)
    assert all(record["exact"] for record in oracle if record["refining"])


def test_run_evaluation_writes_records_and_summary(trained_run):
    root, run = trained_run
    summary = run_evaluation(run, root, "test")

    assert (run / "test_best_records.csv").read_text().count("\n") == 3 * 9 + 1
    stored = json.loads((run / "test_best_summary.json").read_text())
    assert stored == pytest.approx(summary)
    assert summary["oracle/clean/every/exact_recovery"] == 1
    assert summary["observed/clean/every/exact_recovery"] == 1
