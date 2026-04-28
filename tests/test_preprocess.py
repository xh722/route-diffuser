from __future__ import annotations

import torch

from planner.preprocess import (
    MinMaxNormalizer,
    cos_sin_to_heading,
    global_to_local,
    heading_to_cos_sin,
    local_to_global,
)


def test_heading_roundtrip() -> None:
    heading = torch.tensor([0.0, 0.25, -0.75])
    cos_sin = heading_to_cos_sin(heading)
    reconstructed = cos_sin_to_heading(cos_sin)
    assert torch.allclose(reconstructed, heading, atol=1e-5)


def test_coordinate_roundtrip() -> None:
    points = torch.tensor([[[2.0, 1.0], [3.0, 1.5]]])
    origin = torch.tensor([[1.0, 1.0]])
    yaw = torch.tensor([0.3])
    local = global_to_local(points, origin, yaw)
    restored = local_to_global(local, origin, yaw)
    assert torch.allclose(restored, points, atol=1e-5)


def test_min_max_normalizer_roundtrip() -> None:
    normalizer = MinMaxNormalizer(
        minimum=torch.tensor([-2.0, 0.0]), maximum=torch.tensor([2.0, 10.0])
    )
    values = torch.tensor([[0.0, 5.0]])
    normalized = normalizer.normalize(values)
    restored = normalizer.unnormalize(normalized)
    assert torch.allclose(restored, values, atol=1e-5)
