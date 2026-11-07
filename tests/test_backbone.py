import inspect

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

import letree.model
from letree.component_tree import ComponentTreeKind
from letree.data.graphs import to_graph
from letree.experiments.train import RunSettings, build_model, run_model
from letree.generation import build_generation_sample
from letree.model import FEATURE_COUNT, LetreeModel

if not hasattr(letree.model, "AppearanceBackbone") or not hasattr(
    letree.model, "pool_zones"
):
    pytest.skip(
        "AppearanceBackbone and pool_zones are not written yet", allow_module_level=True
    )

from letree.model import AppearanceBackbone, pool_zones  # noqa: E402

wired = pytest.mark.skipif(
    "backbone" not in inspect.signature(LetreeModel).parameters,
    reason="LetreeModel does not take a backbone yet",
)


def _positive_backbone():
    # Positive weights and biases on a positive image keep every ReLU open, so
    # the gradient is positive wherever an input pixel can reach the output
    backbone = AppearanceBackbone().double()
    with torch.no_grad():
        for parameter in backbone.parameters():
            parameter.fill_(0.01)
    return backbone


@pytest.mark.parametrize("size", [(32, 32), (5, 7), (1, 1)])
def test_backbone_keeps_the_image_size(size) -> None:
    image = torch.rand(3, *size)
    assert AppearanceBackbone()(image).shape == (3, 32, *size)


def test_backbone_is_ten_padded_three_by_three_convolutions() -> None:
    convolutions = [
        m for m in AppearanceBackbone().modules() if isinstance(m, torch.nn.Conv2d)
    ]
    assert len(convolutions) == 10
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convolutions)
    assert convolutions[0].in_channels == 1
    assert all(c.out_channels == 32 for c in convolutions)
    assert not any(
        isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm))
        for m in AppearanceBackbone().modules()
    )


def test_one_output_pixel_reads_exactly_a_21_by_21_window() -> None:
    image = (torch.rand(1, 31, 31, dtype=torch.float64) + 0.5).requires_grad_()
    _positive_backbone()(image)[0, :, 15, 15].sum().backward()

    reached = image.grad[0] != 0
    expected = torch.zeros(31, 31, dtype=torch.bool)
    expected[5:26, 5:26] = True
    torch.testing.assert_close(reached, expected)


def test_zeroed_residual_blocks_pass_the_lifted_features_through() -> None:
    # Convolutions are registered in execution order, so 1 to 8 form the blocks
    backbone = AppearanceBackbone().double()
    convolutions = [m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        for convolution in convolutions[1:9]:
            convolution.weight.zero_()
            convolution.bias.zero_()
    image = torch.rand(2, 6, 5, dtype=torch.float64)

    lifted = torch.relu(convolutions[0](image.unsqueeze(1)))
    expected = convolutions[9](torch.relu(lifted))
    torch.testing.assert_close(backbone(image), expected)


def test_images_of_a_batch_do_not_mix() -> None:
    image = torch.rand(2, 9, 9, dtype=torch.float64)
    backbone = AppearanceBackbone().double()
    changed = image.clone()
    changed[1] = torch.rand(9, 9, dtype=torch.float64)
    torch.testing.assert_close(backbone(image)[0], backbone(changed)[0])


def test_pooling_is_the_mean_over_each_zone_in_a_batch() -> None:
    features = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2)
    # Image 0 has nodes 0 to 2, image 1 nodes 3 and 4
    label_map = torch.tensor([0, 0, 1, 2, 3, 3, 3, 4])

    pooled = pool_zones(features, label_map, 5)

    pixels = features.permute(0, 2, 3, 1).reshape(-1, 2)
    expected = torch.stack([pixels[label_map == node].mean(0) for node in range(5)])
    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(pooled[0], torch.tensor([0.5, 4.5], dtype=torch.float64))
    torch.testing.assert_close(
        pooled[3], torch.tensor([9.0, 13.0], dtype=torch.float64)
    )


def _batch():
    images = (
        np.array([[0, 0, 9, 9], [0, 5, 9, 9], [0, 0, 9, 3]], dtype=np.uint8),
        np.array([[7, 7, 7, 1], [7, 2, 7, 1], [7, 7, 7, 1]], dtype=np.uint8),
    )
    graphs = []
    for image in images:
        sample = build_generation_sample(image, ComponentTreeKind.MAX_TREE)
        graphs.append(to_graph(sample.observed, sample.correction_targets))
    return Batch.from_data_list(graphs)


@wired
def test_model_without_a_backbone_is_unchanged() -> None:
    model = LetreeModel(8)
    assert model.backbone is None
    assert model.encoder.in_features == FEATURE_COUNT
    assert not any(key.startswith("backbone.") for key in model.state_dict())


@wired
def test_backbone_features_reach_the_contraction_head() -> None:
    settings = RunSettings(
        dataset_root="",
        max_epochs=1,
        width=8,
        observed_steps=1,
        correction_steps=1,
        backbone=True,
    )
    model = build_model(settings)
    assert isinstance(model.backbone, AppearanceBackbone)
    assert model.encoder.in_features == FEATURE_COUNT + 32

    output = run_model(model, _batch(), settings, settings.correction_steps)
    output.contraction[-1].sum().backward()
    for name, parameter in model.backbone.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
