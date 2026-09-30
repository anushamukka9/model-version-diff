"""Tests for weight statistics computation and diffing."""

import json

import numpy as np
import pytest

from model_version_diff.weights import (
    _histogram,
    compute_weight_stats,
    diff_weight_stats,
    load_weight_stats,
    write_safetensors,
)


def _hist(arr):
    counts, _ = _histogram(np.asarray(arr, dtype=np.float64).ravel())
    return counts


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


def test_compute_weight_stats_includes_histogram(tmp_path):
    rng = np.random.default_rng(3)
    arr = rng.normal(0, 1, size=(50, 50)).astype(np.float32)
    p = tmp_path / "w.npz"
    np.savez(p, layer=arr)
    stats = compute_weight_stats(p)["tensors"]["layer"]
    assert len(stats["histogram"]) == 16
    assert sum(stats["histogram"]) == arr.size
    assert len(stats["hist_edges"]) == 17


def test_histogram_drift_zero_for_identical_distributions():
    rng = np.random.default_rng(4)
    base = {"histogram": _hist(rng.normal(0, 1, size=10_000))}
    d = diff_weight_stats({"a": {**_stats(), **base}}, {"a": {**_stats(), **base}})
    assert d["changed"][0]["histogram_drift"] == pytest.approx(0.0, abs=1e-9)


def test_histogram_drift_detects_shape_change():
    rng = np.random.default_rng(5)
    # Same mean and std, different shape: gaussian vs bimodal.
    gauss = rng.normal(0, 1, size=20_000)
    bimodal = np.concatenate([rng.normal(-1.5, 0.4, size=10_000),
                              rng.normal(1.5, 0.4, size=10_000)])
    bimodal = (bimodal - bimodal.mean()) / bimodal.std()  # match mean/std
    d = diff_weight_stats(
        {"a": {**_stats(), "histogram": _hist(gauss)}},
        {"a": {**_stats(), "histogram": _hist(bimodal)}},
    )
    rec = d["changed"][0]
    assert rec["histogram_drift"] is not None
    assert rec["histogram_drift"] > 0.3  # clearly different shapes


def test_histogram_drift_none_for_legacy_stats():
    # Stats dumps written before histograms existed keep working.
    rec = diff_weight_stats({"a": _stats()}, {"a": _stats()})["changed"][0]
    assert rec["histogram_drift"] is None


def test_safetensors_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    tensors = {
        "w": rng.normal(0, 1, size=(8, 8)).astype(np.float32),
        "b": rng.normal(0, 0.1, size=(8,)).astype(np.float16),
    }
    p = tmp_path / "m.safetensors"
    write_safetensors(p, tensors)
    stats = compute_weight_stats(p)["tensors"]
    assert set(stats) == {"w", "b"}
    assert stats["w"]["shape"] == [8, 8]
    assert stats["b"]["dtype"] == "float16"
    assert stats["w"]["mean"] == pytest.approx(float(tensors["w"].mean()), rel=1e-5)
    assert len(stats["w"]["histogram"]) == 16


def test_compute_weight_stats_pth_needs_torch(tmp_path):
    torch = pytest.importorskip("torch")
    sd = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.zeros(4)}
    p = tmp_path / "m.pth"
    torch.save(sd, p)
    stats = compute_weight_stats(p)["tensors"]
    assert set(stats) == {"layer.weight", "layer.bias"}
    assert stats["layer.weight"]["shape"] == [4, 4]


def test_compute_weight_stats_pth_without_torch_errors(tmp_path):
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; the no-torch error path is not reachable")
    p = tmp_path / "m.pth"
    p.write_bytes(b"fake")
    with pytest.raises(ValueError, match="needs torch"):
        compute_weight_stats(p)
