"""Tests for config / hyperparameter diffing."""

from model_version_diff import diff_configs


def test_added_removed_changed():
    changes = diff_configs({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    by_path = {c.path: c for c in changes}
    assert set(by_path) == {"b", "c"}
    assert by_path["b"].change_type == "changed"
    assert by_path["c"].change_type == "added"
    assert by_path["b"].old == 2 and by_path["b"].new == 3


def test_nested_flattening():
    changes = diff_configs(
        {"training": {"optimizer": {"lr": 0.01}}},
        {"training": {"optimizer": {"lr": 0.001}}},
    )
    assert len(changes) == 1
    assert changes[0].path == "training.optimizer.lr"


def test_semantic_categories():
    changes = diff_configs(
        {"learning_rate": 0.01, "hidden_size": 64},
        {"learning_rate": 0.001, "hidden_size": 128},
    )
    cats = {c.path: c.category for c in changes}
    assert cats["learning_rate"] == "hyperparameter"
    assert cats["hidden_size"] == "architecture"


def test_severity_rules():
    arch_change = diff_configs({"hidden_size": 64}, {"hidden_size": 128})[0]
    assert arch_change.severity == "breaking"
    hp_change = diff_configs({"learning_rate": 0.01}, {"learning_rate": 0.001})[0]
    assert hp_change.severity == "significant"
    removed_arch = diff_configs({"num_layers": 4}, {})[0]
    assert removed_arch.severity == "breaking"
    added_infra = diff_configs({}, {"num_gpus": 8})[0]
    assert added_infra.severity == "minor"


def test_identical_configs_produce_no_changes():
    cfg = {"a": {"b": [1, 2, 3]}, "c": "x"}
    assert diff_configs(cfg, cfg) == []
