"""Quickstart example: diff two versions of a (synthetic) fraud-detection model.

Run from the repo root::

    pip install -e .
    python examples/quickstart.py

This builds two version manifests in a temp directory — v1.0 and v2.0 of a
small MLP — then runs the full four-axis diff and prints the Markdown report.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from model_version_diff import diff_versions, load_manifest, render_markdown
from model_version_diff.weights import compute_weight_stats


def make_weights(path: Path, seed: int, drift: float = 0.0) -> None:
    rng = np.random.default_rng(seed)
    tensors = {
        "encoder.weight": rng.normal(0, 0.5, size=(64, 32)).astype(np.float32),
        "encoder.bias": rng.normal(0, 0.1, size=(64,)).astype(np.float32),
        "head.weight": rng.normal(0, 0.3, size=(2, 64)).astype(np.float32) + drift,
        "head.bias": rng.normal(0, 0.05, size=(2,)).astype(np.float32),
    }
    np.savez(path, **tensors)


def make_predictions(n: int, seed: int, flip_every: int) -> dict:
    rng = np.random.default_rng(seed)
    preds = {}
    labels = ["legit", "fraud"]
    for i in range(n):
        label = labels[int(rng.integers(0, 2))]
        score = round(float(rng.uniform(0.55, 0.99)), 3)
        if flip_every and i % flip_every == 0:
            label = labels[1 - labels.index(label)]
            score = round(float(rng.uniform(0.51, 0.7)), 3)
        preds[f"probe-{i:03d}"] = {"label": label, "score": score}
    return preds


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mvd-quickstart-"))

    make_weights(tmp / "v1.npz", seed=1)
    make_weights(tmp / "v2.npz", seed=1, drift=0.4)  # head.weight shifted on purpose

    stats_v1 = compute_weight_stats(tmp / "v1.npz")
    stats_v2 = compute_weight_stats(tmp / "v2.npz")
    (tmp / "v1_weight_stats.json").write_text(json.dumps(stats_v1))
    (tmp / "v2_weight_stats.json").write_text(json.dumps(stats_v2))

    (tmp / "v1_predictions.json").write_text(json.dumps(make_predictions(50, seed=7, flip_every=0)))
    (tmp / "v2_predictions.json").write_text(json.dumps(make_predictions(50, seed=7, flip_every=10)))

    def manifest(version, **kw):
        doc = {
            "model": "fraud-detector",
            "version": version,
            "config": kw["config"],
            "weights": kw["weights"],
            "dataset": kw["dataset"],
            "predictions": kw["predictions"],
        }
        p = tmp / f"{version}.json"
        p.write_text(json.dumps(doc, indent=2))
        return p

    v1 = manifest(
        "v1.0.0",
        config={"hidden_size": 64, "num_layers": 2, "learning_rate": 0.001,
                "batch_size": 256, "optimizer": "adam", "dropout": 0.1},
        weights="v1_weight_stats.json",
        dataset={"name": "fraud-train", "n_samples": 100_000, "fingerprint": "sha256:aaa",
                 "columns": {"amount": {"type": "numeric", "mean": 42.0, "std": 18.0, "min": 0, "max": 999},
                             "label": {"type": "categorical", "cardinality": 2,
                                       "top_values": ["legit", "fraud"]}}},
        predictions="v1_predictions.json",
    )
    v2 = manifest(
        "v2.0.0",
        config={"hidden_size": 128, "num_layers": 2, "learning_rate": 0.0005,
                "batch_size": 256, "optimizer": "adamw", "dropout": 0.1,
                "label_smoothing": 0.05},
        weights="v2_weight_stats.json",
        dataset={"name": "fraud-train", "n_samples": 132_000, "fingerprint": "sha256:bbb",
                 "columns": {"amount": {"type": "numeric", "mean": 47.5, "std": 21.0, "min": 0, "max": 1200},
                             "merchant": {"type": "categorical", "cardinality": 48,
                                          "top_values": ["grocery", "gas", "online"]},
                             "label": {"type": "categorical", "cardinality": 2,
                                       "top_values": ["legit", "fraud"]}}},
        predictions="v2_predictions.json",
    )

    diff = diff_versions(load_manifest(v1), load_manifest(v2))
    report = render_markdown(diff)
    (tmp / "diff_report.md").write_text(report)
    print(report)
    print(f"\nArtifacts written to {tmp}")


if __name__ == "__main__":
    main()
