"""Tests for weight statistics computation and diffing."""

import json

import numpy as np
import pytest

from model_version_diff.weights import (
    compute_weight_stats,
    diff_weight_stats,
    load_weight_stats,
)


def _stats(mean=0.0, std=1.0, norm=10.0):
    return {"shape": [10], "dtype": "float32", "mean": mean, "std": std, "norm": norm}


def test_compute_weight_stats_npz(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.normal(2.0, 0.5, size=(64, 64)).astype(np.float32)
    p = tmp_path / "w.npz"
    np.savez(p, layer=arr)
    stats = compute_weight_stats(p)["tensors"]["layer"]
    assert stats["shape"] == [64, 64]
    assert stats["mean"] == pytest.approx(2.0, abs=0.05)
    assert stats["std"] == pytest.approx(0.5, abs=0.05)
    assert stats["norm"] == pytest.approx(float(np.linalg.norm(arr)), rel=1e-6)


def test_compute_weight_stats_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match="Unsupported weight file"):
        compute_weight_stats(p)


def test_diff_weight_stats_added_removed_changed():
    old = {"a": _stats(mean=0.0), "b": _stats(mean=1.0)}
    new = {"a": _stats(mean=0.5), "c": _stats(mean=2.0)}
    d = diff_weight_stats(old, new)
    assert d["added"] == ["c"]
    assert d["removed"] == ["b"]
    assert [r["layer"] for r in d["changed"]] == ["a"]
    rec = d["changed"][0]
    assert rec["mean_delta"] == pytest.approx(0.5)
    assert rec["drift_score"] > 0


def test_diff_weight_stats_detects_shape_change():
    old = {"a": {**_stats(), "shape": [4, 4]}}
    new = {"a": {**_stats(), "shape": [4, 8]}}
    rec = diff_weight_stats(old, new)["changed"][0]
    assert rec["shape_changed"] is True


def test_diff_weight_stats_sorted_by_drift():
    old = {"x": _stats(mean=0.0), "y": _stats(mean=0.0)}
    new = {"x": _stats(mean=0.01), "y": _stats(mean=5.0)}
    layers = [r["layer"] for r in diff_weight_stats(old, new)["changed"]]
    assert layers[0] == "y"


def test_load_weight_stats_json_roundtrip(tmp_path):
    payload = {"tensors": {"a": _stats()}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(payload))
    assert load_weight_stats(p) == payload
