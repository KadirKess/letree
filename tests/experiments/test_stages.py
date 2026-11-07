import json
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Batch

from letree.data.graphs import to_graph
from letree.experiments.train import (
    FROZEN_MODULES,
    SELECTION_METRIC,
    RunSettings,
    build_model,
    epoch_report,
    fit,
)
from letree.generation import SaltAndPepperNoise
from letree.model import LetreeModel

NOISES = (SaltAndPepperNoise(probability=0.05),)
CORRECTIVE_MODULES = (
    "corrective_gru",
    "corrective_query",
    "corrective_key",
    "increment_hidden",
    "increment_output",
    "contraction_hidden",
    "contraction_output",
)


@pytest.fixture
def dataset_root(make_dataset):
    return make_dataset({"train": 3, "val": 1}, NOISES, 2, base_seed=3)


def _settings(root, max_epochs, **changes):
    settings = RunSettings(
        dataset_root=str(root),
        max_epochs=max_epochs,
        width=16,
        observed_steps=2,
        correction_steps=1,
        batch_size=4,
    )
    return replace(settings, **changes)


def _module_state(state: dict, module: str) -> dict:
    return {k: v for k, v in state.items() if k.split(".")[0] == module}


def _same(first: dict, second: dict) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[k], second[k]) for k in first
    )


def test_skipping_the_corrective_loop_leaves_the_observed_loop_unchanged(
    square_family,
) -> None:
    family = square_family(1, 0, NOISES, 1)
    batch = Batch.from_data_list([to_graph(o.tree_image, o.targets) for o in family])
    torch.manual_seed(0)
    model = LetreeModel(16)
    arguments = (batch.x, batch.edge_index, batch.rag_edge_index, batch.ptr, 2)

    skipped = model(*arguments, None)
    full = model(*arguments, 1)

    assert skipped.corrective == skipped.increment == skipped.contraction == []
    assert len(skipped.observed) == 2
    for alone, together in zip(skipped.observed, full.observed):
        assert torch.equal(alone.query, together.query)
        assert torch.equal(alone.log_z, together.log_z)


@pytest.fixture
def observed_run(dataset_root, tmp_path):
    run = tmp_path / "observed"
    fit(_settings(dataset_root, 2, stage="observed"), run)
    return run


def test_observed_stage_trains_and_selects_on_the_observed_loss_only(
    observed_run, dataset_root
) -> None:
    history = json.loads((observed_run / "history.json").read_text())
    for entry in history:
        assert set(entry["train"]) == {"observed_parent"}
        validation = entry["validation"]
        assert validation[SELECTION_METRIC] == validation["loss/observed_parent"]
        assert "all/observed_parent_accuracy/image_mean" in validation
        assert "all/set_pointer_accuracy/image_mean" not in validation
        assert "loss/set_pointer" not in validation

    recorded = json.loads((observed_run / "settings.json").read_text())
    assert recorded["settings"]["stage"] == "observed"
    assert recorded["selection_parts"] == ["loss/observed_parent"]

    # No corrective module gets a gradient, so they keep their initialisation
    trained = torch.load(observed_run / "best.pt", map_location="cpu")["model"]
    torch.manual_seed(0)
    initial = build_model(_settings(dataset_root, 2, stage="observed")).state_dict()
    for module in CORRECTIVE_MODULES:
        assert _same(_module_state(trained, module), _module_state(initial, module))
    assert not _same(
        _module_state(trained, "observed_gru"), _module_state(initial, "observed_gru")
    )


def test_frozen_stage_keeps_the_observed_part_and_trains_the_rest(
    observed_run, dataset_root, tmp_path
) -> None:
    observed_best = torch.load(observed_run / "best.pt", map_location="cpu")["model"]
    settings = _settings(
        dataset_root, 1, stage="frozen", observed_run=str(observed_run)
    )
    run = tmp_path / "frozen"

    fit(settings, run)
    fit(replace(settings, max_epochs=2), run, resume=True)

    last = torch.load(run / "last.pt", map_location="cpu")
    for module in FROZEN_MODULES:
        assert _same(
            _module_state(last["model"], module),
            _module_state(observed_best, module),
        ), module
    for module in CORRECTIVE_MODULES:
        assert not _same(
            _module_state(last["model"], module),
            _module_state(observed_best, module),
        ), module

    history = json.loads((run / "history.json").read_text())
    assert [entry["epoch"] for entry in history] == [0, 1]
    observed_losses = [entry["validation"]["loss/observed_parent"] for entry in history]
    assert observed_losses[0] == pytest.approx(observed_losses[1])
    validation = history[-1]["validation"]
    assert validation[SELECTION_METRIC] == pytest.approx(
        validation["loss/set_pointer"]
        + validation["loss/contraction"]
        + validation["loss/increment"]
    )


def test_frozen_stage_rejects_an_observed_run_of_another_shape(
    observed_run, dataset_root, tmp_path
) -> None:
    for change in ({"width": 8}, {"observed_steps": 3}, {"aggregations": ("max",)}):
        settings = _settings(
            dataset_root, 1, stage="frozen", observed_run=str(observed_run), **change
        )
        with pytest.raises(ValueError):
            fit(settings, tmp_path / "frozen")


def test_frozen_stage_rejects_a_run_that_is_not_an_observed_stage(
    dataset_root, tmp_path
) -> None:
    joint = tmp_path / "joint"
    fit(_settings(dataset_root, 1), joint)
    settings = _settings(dataset_root, 1, stage="frozen", observed_run=str(joint))
    with pytest.raises(ValueError, match="not an observed stage"):
        fit(settings, tmp_path / "frozen")


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "unknown"},
        {"stage": "frozen"},
        {"stage": "joint", "observed_run": "somewhere"},
        {"stage": "observed", "observed_run": "somewhere"},
    ],
)
def test_stage_and_observed_run_must_agree(dataset_root, tmp_path, changes) -> None:
    with pytest.raises(ValueError):
        fit(_settings(dataset_root, 1, **changes), tmp_path / "run")


def test_epoch_report_of_an_observed_stage_has_only_observed_columns(
    observed_run,
) -> None:
    history = json.loads((observed_run / "history.json").read_text())
    report = epoch_report(history, 0, 20)
    header = next(line for line in report.splitlines() if "loss" in line)
    assert "observed" in header
    assert "pointer" not in report
    assert "contraction" not in report
    assert "selection (val observed loss)" in report


def test_frozen_stage_loads_an_observed_run_recorded_before_the_backbone_setting(
    observed_run, dataset_root, tmp_path
) -> None:
    path = observed_run / "settings.json"
    recorded = json.loads(path.read_text())
    del recorded["settings"]["backbone"]
    path.write_text(json.dumps(recorded))

    settings = _settings(
        dataset_root, 1, stage="frozen", observed_run=str(observed_run)
    )
    fit(settings, tmp_path / "frozen")
