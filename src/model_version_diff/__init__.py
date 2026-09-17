"""model-version-diff: diff two model versions and explain what changed.

Compare model versions across four axes:

- config / hyperparameters (added / removed / changed keys, semantically typed)
- weight statistics (per-layer mean/std/norm deltas from safetensors, npz, or JSON dumps)
- training-dataset fingerprints (schema, volume, and distribution drift)
- behavior (prediction flips on a probe set)

The primary entry point is :func:`diff_versions`, which takes two version
manifests and returns a structured :class:`VersionDiff` that can be rendered
as Markdown, HTML, or JSON.
"""

from .config_diff import diff_configs, ConfigChange
from .weights import compute_weight_stats, load_weight_stats, diff_weight_stats
from .dataset import diff_datasets, DatasetDiff
from .behavior import diff_predictions, BehaviorDiff
from .manifest import load_manifest, VersionManifest
from .diff import diff_versions, VersionDiff
from .report import render_markdown, render_html

__version__ = "0.1.0"
__author__ = "Anusha Mukka"
__all__ = [
    "diff_versions",
    "diff_configs",
    "diff_weight_stats",
    "diff_datasets",
    "diff_predictions",
    "compute_weight_stats",
    "load_weight_stats",
    "load_manifest",
    "render_markdown",
    "render_html",
    "ConfigChange",
    "DatasetDiff",
    "BehaviorDiff",
    "VersionManifest",
    "VersionDiff",
]
