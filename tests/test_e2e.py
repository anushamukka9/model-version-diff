"""End-to-end tests: manifests, the orchestrator, reports, and the CLI."""

import json

import pytest

from model_version_diff import diff_versions, load_manifest, render_html, render_markdown
from model_version_diff.cli import main as cli_main


def _manifest_doc(tmp_path, version, **kw):
    doc = {"model": "demo", "version": version}
    doc.update(kw)
    p = tmp_path / f"{version}.json"
    p.write_text(json.dumps(doc))
    return p


@pytest.fixture
def two_manifests(tmp_path):
    old = _manifest_doc(
        tmp_path, "v1",
        config={"lr": 0.01, "hidden": 64},
        weights={"tensors": {"w": {"shape": [4], "dtype": "float32",
                                   "mean": 0.0, "std": 1.0, "norm": 2.0}}},
        dataset={"name": "d", "n_samples": 100, "fingerprint": "x", "columns": {}},
        predictions={"p1": "a", "p2": "b"},
    )
    new = _manifest_doc(
        tmp_path, "v2",
        config={"lr": 0.001, "hidden": 64},
        weights={"tensors": {"w": {"shape": [4], "dtype": "float32",
                                   "mean": 0.5, "std": 1.0, "norm": 2.0}}},
        dataset={"name": "d", "n_samples": 120, "fingerprint": "y", "columns": {}},
        predictions={"p1": "b", "p2": "b"},
    )
    return old, new


def test_manifest_file_reference_resolution(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"lr": 0.01}))
    doc = {"model": "m", "version": "v1", "config": {"$file": "config.json"}}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    assert load_manifest(p).config == {"lr": 0.01}


def test_diff_versions_verdict(two_manifests):
    d = diff_versions(*two_manifests)
    assert d.old_version == "v1"
    assert d.new_version == "v2"
    assert len(d.config_changes) == 1
    assert d.behavior_diff.n_flips == 1
    assert "flip" in d.verdict()


def test_render_markdown_contains_sections(two_manifests):
    md = render_markdown(diff_versions(*two_manifests))
    assert "# Model Version Diff" in md
    for section in ("Configuration", "Weight", "dataset", "probe"):
        assert section.lower() in md.lower()


def test_render_html_is_standalone(two_manifests):
    page = render_html(diff_versions(*two_manifests))
    assert page.startswith("<!DOCTYPE html>")
    assert "<table>" in page
    assert "v1" in page and "v2" in page


def test_cli_diff_markdown_to_stdout(two_manifests, capsys):
    assert cli_main(["diff", str(two_manifests[0]), str(two_manifests[1])]) == 0
    out = capsys.readouterr().out
    assert "Model Version Diff" in out


def test_cli_diff_json_format(two_manifests, capsys):
    assert cli_main(["diff", str(two_manifests[0]), str(two_manifests[1]),
                     "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["old_version"] == "v1"
    assert payload["new_version"] == "v2"


def test_cli_diff_writes_out_file(two_manifests, tmp_path):
    out = tmp_path / "report.md"
    assert cli_main(["diff", str(two_manifests[0]), str(two_manifests[1]),
                     "--out", str(out)]) == 0
    assert out.read_text().startswith("# Model Version Diff")


def test_cli_weights_subcommand(tmp_path, capsys):
    import numpy as np
    p = tmp_path / "w.npz"
    np.savez(p, a=np.ones((2, 2), dtype=np.float32))
    assert cli_main(["weights", str(p)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tensors"]["a"]["mean"] == 1.0
