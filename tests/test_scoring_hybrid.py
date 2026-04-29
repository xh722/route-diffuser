from __future__ import annotations

import torch

from planner.datasets import SyntheticDatasetConfig, SyntheticPlanningDataset, collate_scene_batches
from planner.inference.scoring import (
    score_trajectory_candidates,
    score_trajectory_candidates_for_mode,
)


def test_hybrid_scoring_uses_learned_preferences_when_heuristic_is_tied() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=1))
    batch = collate_scene_batches([dataset[0]])

    candidate = batch.future_ego_trajectory.clone()
    predictions = torch.stack([candidate, candidate], dim=1)
    learned_scores = torch.tensor([[0.0, 2.0]], dtype=predictions.dtype)

    scored = score_trajectory_candidates(
        predicted_samples=predictions,
        scene_batch=batch,
        learned_scores=learned_scores,
        learned_weight=1.0,
    )

    assert int(scored["selected_indices"][0].item()) == 1
    assert scored["normalized_learned_scores"] is not None


class _FakeModel:
    class Config:
        learned_scorer_weight = 1.0

    def __init__(self) -> None:
        self.config = self.Config()

    def score_trajectories(self, scene_batch, predicted_samples):
        del scene_batch, predicted_samples
        return torch.tensor([[0.0, 2.0]])


def test_selection_mode_auto_uses_hybrid_when_weight_enabled() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=1))
    batch = collate_scene_batches([dataset[0]])

    candidate = batch.future_ego_trajectory.clone()
    predictions = torch.stack([candidate, candidate], dim=1)
    scored = score_trajectory_candidates_for_mode(
        predicted_samples=predictions,
        scene_batch=batch,
        model=_FakeModel(),
        selection_mode="auto",
    )

    assert int(scored["selected_indices"][0].item()) == 1
