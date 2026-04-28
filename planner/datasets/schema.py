"""Canonical planner-facing scene tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CanonicalSceneBatch:
    """Canonical batched tensors consumed by the planner."""

    ego_current_state: torch.Tensor
    neighbor_history: torch.Tensor
    neighbor_history_mask: torch.Tensor
    lane_polylines: torch.Tensor
    lane_polylines_mask: torch.Tensor
    route_lanes: torch.Tensor
    route_lanes_mask: torch.Tensor
    future_ego_trajectory: torch.Tensor | None = None
    future_ego_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "CanonicalSceneBatch":
        """Validate tensor shapes for the canonical scene schema."""

        batch_size = self.ego_current_state.shape[0]
        self._expect_rank("ego_current_state", self.ego_current_state, 2)
        self._expect_batch("neighbor_history", self.neighbor_history, batch_size)
        self._expect_batch(
            "neighbor_history_mask", self.neighbor_history_mask, batch_size
        )
        self._expect_batch("lane_polylines", self.lane_polylines, batch_size)
        self._expect_batch("lane_polylines_mask", self.lane_polylines_mask, batch_size)
        self._expect_batch("route_lanes", self.route_lanes, batch_size)
        self._expect_batch("route_lanes_mask", self.route_lanes_mask, batch_size)

        self._expect_rank("neighbor_history", self.neighbor_history, 4)
        self._expect_rank("neighbor_history_mask", self.neighbor_history_mask, 3)
        self._expect_rank("lane_polylines", self.lane_polylines, 4)
        self._expect_rank("lane_polylines_mask", self.lane_polylines_mask, 3)
        self._expect_rank("route_lanes", self.route_lanes, 4)
        self._expect_rank("route_lanes_mask", self.route_lanes_mask, 3)

        if self.neighbor_history.shape[:-1] != self.neighbor_history_mask.shape:
            raise ValueError("neighbor_history and neighbor_history_mask mismatch")
        if self.lane_polylines.shape[:-1] != self.lane_polylines_mask.shape:
            raise ValueError("lane_polylines and lane_polylines_mask mismatch")
        if self.route_lanes.shape[:-1] != self.route_lanes_mask.shape:
            raise ValueError("route_lanes and route_lanes_mask mismatch")

        if self.future_ego_trajectory is not None:
            self._expect_batch("future_ego_trajectory", self.future_ego_trajectory, batch_size)
            self._expect_rank("future_ego_trajectory", self.future_ego_trajectory, 3)
            if self.future_ego_trajectory.shape[-1] != self.ego_current_state.shape[-1]:
                raise ValueError("future_ego_trajectory and ego_current_state dim mismatch")
            if self.future_ego_mask is not None:
                self._expect_batch("future_ego_mask", self.future_ego_mask, batch_size)
                self._expect_rank("future_ego_mask", self.future_ego_mask, 2)
                if self.future_ego_trajectory.shape[:-1] != self.future_ego_mask.shape:
                    raise ValueError("future_ego_trajectory and future_ego_mask mismatch")
        elif self.future_ego_mask is not None:
            raise ValueError("future_ego_mask requires future_ego_trajectory")

        return self

    @property
    def batch_size(self) -> int:
        return int(self.ego_current_state.shape[0])

    @property
    def trajectory_dim(self) -> int:
        return int(self.ego_current_state.shape[-1])

    @property
    def future_horizon(self) -> int:
        if self.future_ego_trajectory is None:
            raise ValueError("future_ego_trajectory is not available")
        return int(self.future_ego_trajectory.shape[1])

    def to(self, device: torch.device | str) -> "CanonicalSceneBatch":
        """Move tensor fields to the target device."""

        return CanonicalSceneBatch(
            ego_current_state=self.ego_current_state.to(device),
            neighbor_history=self.neighbor_history.to(device),
            neighbor_history_mask=self.neighbor_history_mask.to(device),
            lane_polylines=self.lane_polylines.to(device),
            lane_polylines_mask=self.lane_polylines_mask.to(device),
            route_lanes=self.route_lanes.to(device),
            route_lanes_mask=self.route_lanes_mask.to(device),
            future_ego_trajectory=(
                None
                if self.future_ego_trajectory is None
                else self.future_ego_trajectory.to(device)
            ),
            future_ego_mask=(
                None if self.future_ego_mask is None else self.future_ego_mask.to(device)
            ),
            metadata=self.metadata,
        ).validate()

    @staticmethod
    def _expect_batch(name: str, tensor: torch.Tensor, batch_size: int) -> None:
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch mismatch: {tensor.shape[0]} != {batch_size}")

    @staticmethod
    def _expect_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
        if tensor.ndim != rank:
            raise ValueError(f"{name} rank mismatch: {tensor.ndim} != {rank}")
