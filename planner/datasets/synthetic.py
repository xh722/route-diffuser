"""Structured synthetic driving scenes used for demos, smoke tests, and debugging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from planner.datasets.schema import CanonicalSceneBatch
from planner.preprocess.angles import heading_to_cos_sin

SCENARIO_SEQUENCE = (
    "keep_lane",
    "lane_change_left",
    "lane_change_right",
    "gentle_curve",
)


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Configuration for the synthetic planning dataset."""

    num_samples: int = 64
    history_steps: int = 8
    future_horizon: int = 16
    state_dim: int = 6
    neighbor_count: int = 6
    lane_count: int = 8
    route_count: int = 2
    polyline_points: int = 32
    lane_dim: int = 4
    lane_width: float = 3.6
    longitudinal_step: float = 2.5
    time_delta: float = 1.0 / 3.0
    seed: int = 7

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SyntheticDatasetConfig":
        filtered = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        return cls(**filtered)


class SyntheticPlanningDataset(Dataset[dict[str, torch.Tensor | int | str]]):
    """Synthetic dataset with structured scenarios for planner bring-up."""

    def __init__(self, config: SyntheticDatasetConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        generator = torch.Generator().manual_seed(self.config.seed + index)
        scenario_name = SCENARIO_SEQUENCE[index % len(SCENARIO_SEQUENCE)]

        future_ego = self._make_future_trajectory(scenario_name, generator)
        ego_state = future_ego[0]

        neighbor_history, neighbor_history_mask = self._make_neighbor_history(
            scenario_name=scenario_name,
            generator=generator,
        )
        lane_polylines, lane_polylines_mask = self._make_lane_polylines(
            scenario_name=scenario_name
        )
        route_lanes, route_lanes_mask = self._make_route_lanes(
            scenario_name=scenario_name
        )
        future_ego_mask = torch.ones(self.config.future_horizon, dtype=torch.bool)

        return {
            "ego_current_state": ego_state,
            "neighbor_history": neighbor_history,
            "neighbor_history_mask": neighbor_history_mask,
            "lane_polylines": lane_polylines,
            "lane_polylines_mask": lane_polylines_mask,
            "route_lanes": route_lanes,
            "route_lanes_mask": route_lanes_mask,
            "future_ego_trajectory": future_ego,
            "future_ego_mask": future_ego_mask,
            "index": index,
            "scenario_name": scenario_name,
        }

    def _make_future_trajectory(
        self, scenario_name: str, generator: torch.Generator
    ) -> torch.Tensor:
        x = torch.arange(self.config.future_horizon, dtype=torch.float32)
        x = x * self.config.longitudinal_step
        y, slope = self._route_profile(x, scenario_name)

        speed = 7.5 + 0.4 * torch.randn(self.config.future_horizon, generator=generator)
        speed = speed.clamp(min=5.5)
        tangent = self._unit_tangent(slope)
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        heading_feature = heading_to_cos_sin(heading)
        velocity = tangent * speed.unsqueeze(-1)

        future = torch.zeros(self.config.future_horizon, self.config.state_dim)
        future[:, 0] = x
        future[:, 1] = y
        future[:, 2:4] = heading_feature
        future[:, 4:6] = velocity
        return future

    def _make_neighbor_history(
        self,
        scenario_name: str,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.zeros(
            self.config.neighbor_count,
            self.config.history_steps,
            self.config.state_dim,
        )
        mask = torch.zeros(
            self.config.neighbor_count, self.config.history_steps, dtype=torch.bool
        )

        active_neighbors = min(self.config.neighbor_count, 4)
        longitudinal_centers = torch.tensor([18.0, 10.0, 2.0, -8.0], dtype=torch.float32)
        lane_offsets = torch.tensor(
            [0.0, self.config.lane_width, -self.config.lane_width, 0.0],
            dtype=torch.float32,
        )
        speeds = torch.tensor([6.5, 7.2, 6.8, 8.4], dtype=torch.float32)

        history_step = self.config.longitudinal_step * 0.8
        for neighbor_idx in range(active_neighbors):
            x = torch.arange(self.config.history_steps, dtype=torch.float32)
            x = longitudinal_centers[neighbor_idx] - history_step * (
                self.config.history_steps - 1 - x
            )
            x = x + 0.3 * torch.randn(self.config.history_steps, generator=generator)
            y, slope = self._lane_profile(
                x=x,
                lane_offset=float(lane_offsets[neighbor_idx].item()),
                scenario_name=scenario_name,
            )
            tangent = self._unit_tangent(slope)
            heading = torch.atan2(tangent[:, 1], tangent[:, 0])
            velocity = tangent * speeds[neighbor_idx]

            history[neighbor_idx, :, 0] = x
            history[neighbor_idx, :, 1] = y
            history[neighbor_idx, :, 2:4] = heading_to_cos_sin(heading)
            history[neighbor_idx, :, 4:6] = velocity
            mask[neighbor_idx] = True

        if active_neighbors > 0 and self.config.history_steps > 2:
            mask[active_neighbors - 1, -2:] = False

        return history, mask

    def _make_lane_polylines(self, scenario_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        polylines = torch.zeros(
            self.config.lane_count, self.config.polyline_points, self.config.lane_dim
        )
        mask = torch.zeros(
            self.config.lane_count, self.config.polyline_points, dtype=torch.bool
        )

        active_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
        x = torch.linspace(
            0.0,
            (self.config.future_horizon + 6) * self.config.longitudinal_step,
            self.config.polyline_points,
        )

        for lane_idx, offset_scale in enumerate(active_offsets[: self.config.lane_count]):
            lane_offset = offset_scale * self.config.lane_width
            y, slope = self._lane_profile(x, lane_offset=lane_offset, scenario_name=scenario_name)
            polylines[lane_idx] = self._pack_polyline_features(x, y, slope)
            mask[lane_idx] = True

        return polylines, mask

    def _make_route_lanes(self, scenario_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        polylines = torch.zeros(
            self.config.route_count, self.config.polyline_points, self.config.lane_dim
        )
        mask = torch.zeros(
            self.config.route_count, self.config.polyline_points, dtype=torch.bool
        )

        x = torch.linspace(
            0.0,
            (self.config.future_horizon + 4) * self.config.longitudinal_step,
            self.config.polyline_points,
        )
        y, slope = self._route_profile(x, scenario_name)
        polylines[0] = self._pack_polyline_features(x, y, slope)
        mask[0] = True

        if self.config.route_count > 1:
            shoulder_y = y + 0.5 * self.config.lane_width
            polylines[1] = self._pack_polyline_features(x, shoulder_y, slope)

        return polylines, mask

    def _route_profile(
        self, x: torch.Tensor, scenario_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(x, dtype=torch.float32)
        max_x = max(float(x.max().item()), 1.0)

        if scenario_name == "keep_lane":
            y = torch.zeros_like(x)
            slope = torch.zeros_like(x)
        elif scenario_name == "lane_change_left":
            transition = torch.sigmoid((x - 0.45 * max_x) / (0.12 * max_x))
            y = self.config.lane_width * transition
            slope = self.config.lane_width * transition * (1.0 - transition) / (0.12 * max_x)
        elif scenario_name == "lane_change_right":
            transition = torch.sigmoid((x - 0.45 * max_x) / (0.12 * max_x))
            y = -self.config.lane_width * transition
            slope = -self.config.lane_width * transition * (1.0 - transition) / (0.12 * max_x)
        elif scenario_name == "gentle_curve":
            normalized = x / max_x
            y = 1.8 * self.config.lane_width * normalized.square()
            slope = 3.6 * self.config.lane_width * x / (max_x * max_x)
        else:
            raise ValueError(f"Unknown scenario_name: {scenario_name}")

        return y, slope

    def _lane_profile(
        self,
        x: torch.Tensor,
        lane_offset: float,
        scenario_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(x, dtype=torch.float32)
        if scenario_name == "gentle_curve":
            base_y, slope = self._route_profile(x, scenario_name)
            return base_y + lane_offset, slope
        return torch.full_like(x, lane_offset), torch.zeros_like(x)

    @staticmethod
    def _unit_tangent(slope: torch.Tensor) -> torch.Tensor:
        tangent = torch.stack([torch.ones_like(slope), slope], dim=-1)
        return tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)

    def _pack_polyline_features(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        slope: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.zeros(self.config.polyline_points, self.config.lane_dim)
        heading = torch.atan2(slope, torch.ones_like(slope))
        heading_feature = heading_to_cos_sin(heading)
        features[:, 0] = x
        features[:, 1] = y
        if self.config.lane_dim >= 4:
            features[:, 2:4] = heading_feature
        if self.config.lane_dim > 4:
            features[:, 4:] = 0.0
        return features


def collate_scene_batches(
    samples: list[dict[str, torch.Tensor | int | str]]
) -> CanonicalSceneBatch:
    """Collate synthetic samples into the canonical batched scene schema."""

    batch = CanonicalSceneBatch(
        ego_current_state=torch.stack([sample["ego_current_state"] for sample in samples]),
        neighbor_history=torch.stack([sample["neighbor_history"] for sample in samples]),
        neighbor_history_mask=torch.stack(
            [sample["neighbor_history_mask"] for sample in samples]
        ),
        lane_polylines=torch.stack([sample["lane_polylines"] for sample in samples]),
        lane_polylines_mask=torch.stack(
            [sample["lane_polylines_mask"] for sample in samples]
        ),
        route_lanes=torch.stack([sample["route_lanes"] for sample in samples]),
        route_lanes_mask=torch.stack([sample["route_lanes_mask"] for sample in samples]),
        future_ego_trajectory=torch.stack(
            [sample["future_ego_trajectory"] for sample in samples]
        ),
        future_ego_mask=torch.stack([sample["future_ego_mask"] for sample in samples]),
        metadata={
            "indices": [int(sample["index"]) for sample in samples],
            "scenario_names": [str(sample["scenario_name"]) for sample in samples],
        },
    )
    return batch.validate()
