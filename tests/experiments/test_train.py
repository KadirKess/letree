import json

import pytest
import torch
from torch_geometric.data import Batch

from letree.data.graphs import ObservationDataset, make_loader, to_graph
from letree.experiments.losses import correction_losses
from letree.experiments.train import (
    SELECTION_METRIC,
    RunSettings,
    build_model,
    epoch_report,
    evaluate,
    fit,
)
from letree.generation import (
    NOISE_KIND_NAMES,
    AdditiveGaussianNoise,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    SquareImageConfig,
)
from letree.model import LetreeModel

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)
NOISES = (SaltAndPepperNoise(probability=0.05),)


def test_all_three_losses_fall_on_a_few_graphs(square_family) -> None:
    # A learning check for target indexing and gradient flow, not learnability
    family = square_family(1, 0, NOISES, 2)
    batch = Batch.from_data_list([to_graph(o.tree_image, o.targets) for o in family])
    torch.manual_seed(0)
    model = LetreeModel(16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def losses():
        output = model(batch.x, batch.edge_index, batch.ptr, 2, 1)
        return correction_losses(output, batch)

    initial = {name: loss.item() for name, loss in losses().items()}
    for _ in range(150):
        optimizer.zero_grad()
        sum(losses().values()).backward()
        optimizer.step()
    final = {name: loss.item() for name, loss in losses().items()}

    for name in initial:
        assert final[name] < 0.5 * initial[name], name


@pytest.fixture
def dataset_root(make_dataset):
    # No test split, so a run that reads test fails
    return make_dataset({"train": 3, "val": 1}, NOISES, 2, base_seed=2)


def _settings(root, max_epochs):
    return RunSettings(
        dataset_root=str(root),
        max_epochs=max_epochs,
        width=16,
        observed_steps=2,
        correction_steps=1,
        batch_size=4,
    )


def test_fit_writes_run_records_and_resumes(dataset_root, tmp_path) -> None:
    run = tmp_path / "run"

    fit(_settings(dataset_root, 2), run)
    best = fit(_settings(dataset_root, 3), run, resume=True)

    history = json.loads((run / "history.json").read_text())
    assert [entry["epoch"] for entry in history] == [0, 1, 2]
    settings = json.loads((run / "settings.json").read_text())
    assert settings["settings"]["observed_steps"] == 2
    assert settings["dataset_manifest"]["split_sample_counts"]["train"] == 3
    scores = [entry["validation"][SELECTION_METRIC] for entry in history]
    assert best == history[scores.index(max(scores))]
    assert best["validation"]["all/images"] == 3
    assert {path.name for path in run.iterdir()} == {
        "settings.json",
        "history.json",
        "best.pt",
        "last.pt",
    }


def test_validation_reports_every_noise_kind_on_the_same_images(make_dataset):
    noises = (
        SaltAndPepperNoise(probability=0.05),
        AdditiveGaussianNoise(standard_deviation=10.0),
        SpeckleNoise(),
        PoissonNoise(),
    )
    root = make_dataset({"val": 3}, noises, 4, base_seed=5)
    settings = _settings(root, 1)
    torch.manual_seed(0)
    model = build_model(settings)
    loader = make_loader(ObservationDataset(root, "val", True), 4, False, 0)

    summary = evaluate(model, loader, settings, torch.device("cpu"))

    kinds = [NOISE_KIND_NAMES[type(noise)] for noise in noises]
    assert [summary[f"{kind}/images"] for kind in kinds] == [3, 3, 3, 3]
    assert summary["clean/images"] == 3
    assert summary["noisy/images"] == 12
    assert sum(summary[f"{kind}/nodes"] for kind in kinds) == summary["noisy/nodes"]
    # The noisy image mean is the mean of equally sized kinds
    joint = "joint_correction_accuracy/image_mean"
    assert summary[f"noisy/{joint}"] == pytest.approx(
        sum(summary[f"{kind}/{joint}"] for kind in kinds) / 4
    )


def _history_entry(epoch, score, groups=("clean", "noisy")):
    validation = {
        "loss/observed_parent": 0.5,
        "loss/corrective_parent": 0.25,
        "loss/action": 0.125,
        SELECTION_METRIC: score,
    }
    accuracies = {
        "observed_parent_accuracy": 0.9,
        "corrective_parent_accuracy": 0.8,
        "action_accuracy": 0.7,
        "joint_correction_accuracy": 0.6,
    }
    for group in groups:
        validation[f"{group}/images"] = 1
        for name, value in accuracies.items():
            # Clean values differ from noisy ones so the rows can be told apart
            offset = 0.05 if group == "clean" else 0.0
            validation[f"{group}/{name}/image_mean"] = value + offset
    return {
        "epoch": epoch,
        "seconds": 12.0,
        "learning_rate": 1e-3,
        "train": {"observed_parent": 1.0, "corrective_parent": 2.0, "action": 4.0},
        "validation": validation,
    }


def _row(report, label):
    line = next(line for line in report.splitlines() if line.startswith(f"  {label}"))
    return [float(value) for value in line[len(label) + 2 :].split()]


def test_epoch_report_separates_losses_and_groups() -> None:
    history = [_history_entry(0, 0.4), _history_entry(1, 0.6), _history_entry(2, 0.5)]
    report = epoch_report(history, stale=1, patience=10)

    assert report.splitlines()[0] == "epoch 2  lr 1.0e-03  12s"
    assert _row(report, "train (epoch mean)") == [1.0, 2.0, 4.0, 7.0]
    assert _row(report, "val ") == [0.5, 0.25, 0.125, 0.875]
    assert _row(report, "val clean") == [0.95, 0.85, 0.75, 0.65]
    assert _row(report, "val noisy") == [0.9, 0.8, 0.7, 0.6]
    # Epoch 1 improved, epoch 2 did not
    assert report.splitlines()[-1] == (
        "  selection (val joint, image mean) 0.5000  best 0.6000 at epoch 1  stale 1/10"
    )


def test_epoch_report_of_a_clean_only_run_has_no_noisy_row() -> None:
    report = epoch_report([_history_entry(0, 0.4, groups=("clean",))], 0, 10)
    assert "val clean" in report
    assert "val noisy" not in report


def test_epoch_report_without_a_finite_score() -> None:
    report = epoch_report([_history_entry(0, float("nan"))], stale=1, patience=10)
    assert "best none yet  stale 1/10" in report


def test_resumed_run_matches_an_uninterrupted_one(
    dataset_root, tmp_path, monkeypatch
) -> None:
    # GPU kernels such as scatter can be nondeterministic, so compare on the CPU
    monkeypatch.setattr(
        "letree.experiments.train.select_device", lambda: torch.device("cpu")
    )
    fit(_settings(dataset_root, 2), tmp_path / "whole")
    fit(_settings(dataset_root, 1), tmp_path / "split")
    fit(_settings(dataset_root, 2), tmp_path / "split", resume=True)

    whole = torch.load(tmp_path / "whole" / "last.pt")["model"]
    split = torch.load(tmp_path / "split" / "last.pt")["model"]
    for name, value in whole.items():
        torch.testing.assert_close(split[name], value, rtol=0, atol=0, msg=name)
