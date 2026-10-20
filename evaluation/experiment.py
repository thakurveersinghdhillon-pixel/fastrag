"""FastRAG Experiment Metadata and Reproducibility Tracking.

Provides lightweight, JSON-serializable experiment records linking evaluation results
to exact dataset versions, manifest SHA-256 checksums, and retrieval configurations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMetadata:
    """Lightweight metadata descriptor for evaluation experiments."""

    experiment_id: str
    description: str
    dataset_version: str
    dataset_manifest_checksum: str
    retrieval_configuration: dict[str, Any] = field(default_factory=dict)
    metric_configuration: list[str] = field(default_factory=list)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    evaluation_output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment metadata to a standard dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize experiment metadata to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, filepath: Path | str) -> None:
        """Write experiment metadata to JSON file."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetadata:
        """Instantiate ExperimentMetadata from a dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            description=data.get("description", ""),
            dataset_version=data.get("dataset_version", "unknown"),
            dataset_manifest_checksum=data.get("dataset_manifest_checksum", ""),
            retrieval_configuration=data.get("retrieval_configuration", {}),
            metric_configuration=data.get("metric_configuration", []),
            timestamp_utc=data.get(
                "timestamp_utc", datetime.now(timezone.utc).isoformat()
            ),
            evaluation_output_path=data.get("evaluation_output_path"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentMetadata:
        """Instantiate ExperimentMetadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file for experiment reproducibility tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_experiment_metadata(
    experiment_id: str,
    description: str,
    manifest_path: Path | str,
    retrieval_configuration: dict[str, Any] | None = None,
    metric_configuration: list[str] | None = None,
    evaluation_output_path: str | None = None,
) -> ExperimentMetadata:
    """Construct an ExperimentMetadata record by inspecting dataset manifest file.

    Computes SHA-256 of the manifest file to lock the exact dataset revision.
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Dataset manifest not found at: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_sha256 = compute_file_sha256(manifest_file)
    dataset_version = manifest_data.get("version", "1.0.0")

    return ExperimentMetadata(
        experiment_id=experiment_id,
        description=description,
        dataset_version=dataset_version,
        dataset_manifest_checksum=manifest_sha256,
        retrieval_configuration=retrieval_configuration or {},
        metric_configuration=metric_configuration or ["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"],
        evaluation_output_path=evaluation_output_path,
    )
