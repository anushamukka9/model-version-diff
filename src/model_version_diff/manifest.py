"""Version manifests: how a model version is described to the differ.

A manifest is a small JSON document (plus optional sidecar files) that
describes everything the differ needs to know about one model version:

.. code-block:: json

    {
      "model": "fraud-detector",
      "version": "v2.1.0",
      "config": {"layers": [64, 32], "learning_rate": 0.001, "...": "..."},
      "weights": "v2.1.0/weight_stats.json",
      "dataset": "v2.1.0/dataset_fingerprint.json",
      "predictions": "v2.1.0/probe_predictions.json"
    }

The ``config`` key may be an inline object or ``{"$file": "config.json"}``.
``weights`` accepts a precomputed stats JSON file, a ``.safetensors`` file,
or an ``.npz`` file (stats are computed on the fly from the latter two, so
you never need a full model framework installed). ``dataset`` and
``predictions`` accept inline objects or paths to JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .weights import compute_weight_stats, load_weight_stats


@dataclass
class VersionManifest:
    """A resolved, in-memory description of one model version."""

    model: str
    version: str
    config: dict = field(default_factory=dict)
    weight_stats: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    source: str = ""


def _resolve(value: Any, base: Path) -> Any:
    """Resolve an inline value or a {"$file": path} / bare-path reference."""
    if isinstance(value, Mapping) and set(value.keys()) == {"$file"}:
        path = (base / value["$file"]).expanduser()
        return json.loads(path.read_text())
    if isinstance(value, str) and value:
        path = (base / value).expanduser()
        if path.is_file():
            if path.suffix.lower() in {".safetensors", ".npz"}:
                return load_weight_stats(path)
            return json.loads(path.read_text())
    return value if value is not None else {}


def load_manifest(path: str | Path) -> VersionManifest:
    """Load and fully resolve a version manifest from a JSON file."""
    path = Path(path).expanduser().resolve()
    base = path.parent
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest {path} must be a JSON object")

    weights_raw = _resolve(raw.get("weights"), base)
    if isinstance(weights_raw, dict) and "tensors" not in weights_raw:
        # Already a stats dict (compute_weight_stats output or flat layer map).
        weight_stats = weights_raw
    elif isinstance(weights_raw, dict):
        weight_stats = weights_raw
    else:
        weight_stats = {}

    return VersionManifest(
        model=str(raw.get("model", path.stem)),
        version=str(raw.get("version", "unknown")),
        config=dict(_resolve(raw.get("config"), base) or {}),
        weight_stats=weight_stats,
        dataset=dict(_resolve(raw.get("dataset"), base) or {}),
        predictions=dict(_resolve(raw.get("predictions"), base) or {}),
        source=str(path),
    )


__all__ = ["VersionManifest", "load_manifest", "compute_weight_stats"]
