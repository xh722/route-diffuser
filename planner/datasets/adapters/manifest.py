"""Manifest structures for adapter-backed planning datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


@dataclass
class SceneManifestEntry:
    """One dataset entry addressable by an adapter."""

    entry_id: str
    source_path: str | None = None
    sample_index: int | None = None
    scenario_name: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SceneManifestEntry":
        return cls(
            entry_id=str(values["entry_id"]),
            source_path=None if values.get("source_path") in (None, "") else str(values["source_path"]),
            sample_index=None
            if values.get("sample_index") is None
            else int(values["sample_index"]),
            scenario_name=None
            if values.get("scenario_name") in (None, "")
            else str(values["scenario_name"]),
            tags=tuple(str(tag) for tag in values.get("tags", [])),
            metadata=dict(values.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source_path": self.source_path,
            "sample_index": self.sample_index,
            "scenario_name": self.scenario_name,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass
class DatasetManifest:
    """Serializable dataset manifest used by adapters and preparation scripts."""

    dataset_name: str
    adapter_type: str
    split: str
    scene_count: int
    config: dict[str, Any] = field(default_factory=dict)
    entries: list[SceneManifestEntry] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest_version {self.manifest_version}; expected {MANIFEST_VERSION}"
            )
        if self.scene_count != len(self.entries):
            raise ValueError(
                f"scene_count {self.scene_count} does not match entry count {len(self.entries)}"
            )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DatasetManifest":
        entries = [
            SceneManifestEntry.from_mapping(item)
            for item in values.get("entries", [])
        ]
        scene_count = int(values.get("scene_count", len(entries)))
        return cls(
            manifest_version=int(values.get("manifest_version", MANIFEST_VERSION)),
            dataset_name=str(values["dataset_name"]),
            adapter_type=str(values["adapter_type"]),
            split=str(values.get("split", "train")),
            scene_count=scene_count,
            config=dict(values.get("config", {})),
            entries=entries,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        manifest_path = Path(path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected manifest mapping at {manifest_path}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "dataset_name": self.dataset_name,
            "adapter_type": self.adapter_type,
            "split": self.split,
            "scene_count": self.scene_count,
            "config": self.config,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path: str | Path) -> Path:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return manifest_path
