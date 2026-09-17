"""Tests for dataset fingerprint diffing."""

from model_version_diff import diff_datasets


def _fingerprint(**overrides):
    base = {
        "name": "train",
        "n_samples": 1000,
        "fingerprint": "sha256:abc",
        "columns": {
            "age": {"type": "numeric", "mean": 35.0, "std": 10.0, "min": 18, "max": 80},
            "label": {"type": "categorical", "cardinality": 2, "top_values": ["a", "b"]},
        },
    }
    base.update(overrides)
    return base


def test_identical_fingerprints_match():
    d = diff_datasets(_fingerprint(), _fingerprint())
    assert d.fingerprint_match is True
    assert d.sample_delta_pct == 0.0
    assert d.columns_added == [] and d.columns_removed == []


def test_sample_growth_and_fingerprint_change():
    d = diff_datasets(_fingerprint(), _fingerprint(n_samples=1500, fingerprint="sha256:def"))
    assert d.fingerprint_match is False
    assert d.sample_delta_pct == 50.0


def test_schema_changes_detected():
    old = _fingerprint()
    new = _fingerprint()
    del new["columns"]["age"]
    new["columns"]["income"] = {"type": "numeric", "mean": 50.0, "std": 5.0}
    d = diff_datasets(old, new)
    assert d.columns_removed == ["age"]
    assert d.columns_added == ["income"]


def test_numeric_drift_standardized_shift():
    old = _fingerprint()
    new = _fingerprint()
    new["columns"]["age"]["mean"] = 45.0  # +1 old-std shift
    drift = next(x for x in diff_datasets(old, new).column_drifts if x.column == "age")
    assert drift.kind == "numeric"
    assert drift.detail["standardized_mean_shift"] == 1.0


def test_categorical_drift_new_values():
    old = _fingerprint()
    new = _fingerprint()
    new["columns"]["label"]["top_values"] = ["a", "b", "c"]
    new["columns"]["label"]["cardinality"] = 3
    drift = next(x for x in diff_datasets(old, new).column_drifts if x.column == "label")
    assert drift.kind == "categorical"
    assert drift.detail["new_values"] == ["c"]


def test_missing_fingerprint_gives_none():
    old = _fingerprint()
    del old["fingerprint"]
    assert diff_datasets(old, _fingerprint()).fingerprint_match is None
