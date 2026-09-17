"""Tests for behavior (prediction flip) diffing."""

from model_version_diff import diff_predictions


def test_flip_rate_and_transitions():
    old = {"p1": "a", "p2": "a", "p3": "b", "p4": "b"}
    new = {"p1": "b", "p2": "a", "p3": "b", "p4": "a"}
    d = diff_predictions(old, new)
    assert d.n_common == 4
    assert d.n_flips == 2
    assert d.flip_rate == 0.5
    assert d.transitions == {"a -> b": 1, "b -> a": 1}
    assert {f.probe_id for f in d.flips} == {"p1", "p4"}


def test_scored_predictions_track_confidence_delta():
    old = {"p1": {"label": "a", "score": 0.9}, "p2": {"label": "b", "score": 0.8}}
    new = {"p1": {"label": "a", "score": 0.6}, "p2": {"label": "c", "score": 0.7}}
    d = diff_predictions(old, new)
    assert d.n_flips == 1
    cs = d.confidence_stats
    assert cs["n_scored_agreements"] == 1
    assert cs["mean_delta"] == -0.3


def test_probe_set_skew_reported():
    old = {"p1": "a", "p2": "a"}
    new = {"p2": "a", "p3": "a"}
    d = diff_predictions(old, new)
    assert d.probes_only_old == ["p1"]
    assert d.probes_only_new == ["p3"]
    assert d.n_common == 1


def test_empty_predictions():
    d = diff_predictions({}, {})
    assert d.n_common == 0
    assert d.flip_rate == 0.0
    assert d.n_flips == 0
