"""Orchestration: diff two manifests across all four axes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .behavior import BehaviorDiff, diff_predictions
from .config_diff import ConfigChange, diff_configs
from .dataset import DatasetDiff, diff_datasets
from .manifest import VersionManifest, load_manifest
from .weights import diff_weight_stats


@dataclass
class VersionDiff:
    """The complete diff between two model versions."""

    model: str
    old_version: str
    new_version: str
    config_changes: List[ConfigChange] = field(default_factory=list)
    weight_diff: Dict[str, Any] = field(default_factory=dict)
    dataset_diff: DatasetDiff | None = None
    behavior_diff: BehaviorDiff | None = None

    def verdict(self) -> str:
        """One-line human summary of the diff."""
        parts = []
        breaking = [c for c in self.config_changes if c.severity == "breaking"]
        sig = [c for c in self.config_changes if c.severity == "significant"]
        if breaking:
            parts.append(f"{len(breaking)} breaking config change(s)")
        if sig:
            parts.append(f"{len(sig)} significant config change(s)")
        ws = (self.weight_diff or {}).get("summary", {})
        if ws.get("added") or ws.get("removed"):
            parts.append(f"weights: +{ws['added']}/-{ws['removed']} layers")
        if self.dataset_diff and self.dataset_diff.fingerprint_match is False:
            parts.append("training data changed")
        if self.behavior_diff and self.behavior_diff.n_flips:
            parts.append(
                f"{self.behavior_diff.n_flips}/{self.behavior_diff.n_common} probe flips "
                f"({100 * self.behavior_diff.flip_rate:.1f}%)"
            )
        return "; ".join(parts) if parts else "no material differences detected"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "verdict": self.verdict(),
            "config_changes": [c.as_dict() for c in self.config_changes],
            "weight_diff": self.weight_diff,
            "dataset_diff": self.dataset_diff.as_dict() if self.dataset_diff else None,
            "behavior_diff": self.behavior_diff.as_dict() if self.behavior_diff else None,
        }


def diff_versions(old: VersionManifest | str | Path, new: VersionManifest | str | Path) -> VersionDiff:
    """Diff two versions given as manifests (or paths to manifest JSON files)."""
    old_m = load_manifest(old) if not isinstance(old, VersionManifest) else old
    new_m = load_manifest(new) if not isinstance(new, VersionManifest) else new

    dataset_diff = None
    if old_m.dataset or new_m.dataset:
        dataset_diff = diff_datasets(old_m.dataset, new_m.dataset)

    behavior_diff = None
    if old_m.predictions or new_m.predictions:
        behavior_diff = diff_predictions(old_m.predictions, new_m.predictions)

    return VersionDiff(
        model=new_m.model or old_m.model,
        old_version=old_m.version,
        new_version=new_m.version,
        config_changes=diff_configs(old_m.config, new_m.config),
        weight_diff=diff_weight_stats(old_m.weight_stats, new_m.weight_stats),
        dataset_diff=dataset_diff,
        behavior_diff=behavior_diff,
    )


__all__ = ["VersionDiff", "diff_versions"]
