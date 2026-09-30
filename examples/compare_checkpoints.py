"""Compare two checkpoints end to end through real .safetensors files.

Run from the repo root::

    pip install -e .
    python examples/compare_checkpoints.py

This writes two small .safetensors checkpoints (v1 and a fine-tuned v2) to a
temp directory, computes weight stats with the pure-Python safetensors reader
(no torch needed), diffs them, and prints the weight section of the Markdown
report. v2's output head is deliberately shifted so the drift ranking has
something to find.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from model_version_diff import diff_versions, render_markdown
from model_version_diff.manifest import VersionManifest
from model_version_diff.weights import load_weight_stats, write_safetensors


def make_checkpoint(path: Path, seed: int, head_shift: float = 0.0) -> None:
    rng = np.random.default_rng(seed)
    write_safetensors(
        path,
        {
            "encoder.weight": rng.normal(0.0, 0.4, size=(32, 16)).astype(np.float32),
            "encoder.bias": rng.normal(0.0, 0.1, size=(32,)).astype(np.float32),
            "head.weight": (
                rng.normal(0.0, 0.2, size=(4, 32)).astype(np.float32) + head_shift
            ),
        },
    )


def manifest(version: str, weights: Path) -> VersionManifest:
    return VersionManifest(
        model="tiny-classifier",
        version=version,
        config={"hidden_size": 32, "num_layers": 1},
        weight_stats=load_weight_stats(weights),
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mvd-checkpoints-"))
    v1_path = tmp / "v1.safetensors"
    v2_path = tmp / "v2.safetensors"
    make_checkpoint(v1_path, seed=11)
    make_checkpoint(v2_path, seed=11, head_shift=0.6)

    diff = diff_versions(manifest("v1", v1_path), manifest("v2", v2_path))
    print(render_markdown(diff))
    print(f"\nCheckpoints kept at {tmp}")

    # Show the raw drift records too, so the ranking is inspectable.
    for rec in diff.weight_diff["changed"]:
        hd = rec["histogram_drift"]
        print(
            f"{rec['layer']:<16} drift_score={rec['drift_score']:.4f} "
            f"shape_drift={hd:.3f}" if hd is not None else f"{rec['layer']:<16} no histogram"
        )


if __name__ == "__main__":
    main()
