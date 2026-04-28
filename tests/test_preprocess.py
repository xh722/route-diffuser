from __future__ import annotations

import torch

from planner.diffusion import pyramid_noise_like
from planner.preprocess import (
    MinMaxNormalizer,
    build_route_trajectory_prior,
    cos_sin_to_heading,
    global_to_local,
    heading_to_cos_sin,
    local_to_global,
)
from planner.datasets import SyntheticDatasetConfig, SyntheticPlanningDataset, collate_scene_batches


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


def test_route_trajectory_prior_matches_synthetic_scene() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=2))
    batch = collate_scene_batches([dataset[0], dataset[1]])
    prior = build_route_trajectory_prior(
        scene_batch=batch,
        future_horizon=batch.future_horizon,
        longitudinal_step=2.5,
    )

    xy_error = torch.linalg.norm(
        prior[..., :2] - batch.future_ego_trajectory[..., :2],
        dim=-1,
    )
    assert torch.allclose(prior[:, 0], batch.ego_current_state)
    assert float(xy_error.mean()) < 1.5


def test_pyramid_noise_like_matches_shape_and_is_finite() -> None:
    trajectory = torch.zeros(2, 16, 6)
    noise = pyramid_noise_like(trajectory)

    assert noise.shape == trajectory.shape
    assert torch.isfinite(noise).all()
