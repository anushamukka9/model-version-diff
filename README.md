# model-version-diff

**Diff model versions and explain what changed** — across configs, weights, training data, and behavior.

When you promote a model from `v1` to `v2`, the question is never "did the version
number change" — it's *what* changed and *whether it matters*. `model-version-diff`
compares two model versions across four axes and renders a Markdown or HTML report
you can attach to a release, a model card, or a review thread:

| Axis | What it compares |
|------|------------------|
| ⚙️ **Config** | Nested config/hyperparameter diffs, semantically typed (hyperparameter, architecture, data, infra) with severity: 🔴 breaking, 🟡 significant, ⚪ minor |
| 🧮 **Weights** | Per-layer mean/std/norm deltas from `.safetensors` (pure-Python parsing — no torch needed), `.npz`, or precomputed stats JSON; ranks layers by drift |
| 📦 **Dataset** | Training-dataset fingerprint comparison: volume change, schema changes, standardized distribution drift |
| 🎯 **Behavior** | Prediction flips on a fixed probe set: flip rate, label transitions, confidence drift |

No model framework required at diff time — versions are described by small JSON
**manifests** plus optional sidecar files (stats dumps, prediction JSON).

## Install

```bash
pip install model-version-diff
# or from source:
git clone https://github.com/anushamukka9/model-version-diff
cd model-version-diff
pip install -e .
```

Requires Python ≥ 3.10 and `numpy` (installed automatically).

## Quickstart

Describe each version with a manifest JSON:

```jsonc
// v1.json
{
  "model": "fraud-detector",
  "version": "v1.0.0",
  "config": {"hidden_size": 64, "learning_rate": 0.001},
  "weights": "v1_weight_stats.json",   // or "model.safetensors", "model.npz"
  "dataset": "v1_dataset.json",        // fingerprint descriptor
  "predictions": "v1_probes.json"      // {"probe-001": {"label": "legit", "score": 0.9}}
}
```

Then diff:

```bash
# Markdown report to stdout
model-version-diff diff v1.json v2.json

# HTML report to a file
model-version-diff diff v1.json v2.json --format html --out diff.html

# Machine-readable JSON
model-version-diff diff v1.json v2.json --format json --out diff.json

# Compute weight stats from raw tensors (safetensors needs no torch)
model-version-diff weights model.safetensors --out weight_stats.json
```

Or run the bundled example (builds two synthetic versions and diffs them):

```bash
python examples/quickstart.py
```

## API

```python
from model_version_diff import diff_versions, render_markdown, render_html

diff = diff_versions("v1.json", "v2.json")   # paths, or VersionManifest objects
print(diff.verdict())
# "1 breaking config change(s); training data changed; 24/50 probe flips (48.0%)"

with open("report.md", "w") as f:
    f.write(render_markdown(diff))
```

Lower-level building blocks are also importable directly:
`diff_configs`, `diff_weight_stats`, `diff_datasets`, `diff_predictions`,
`compute_weight_stats`, `load_manifest`.

## Architecture

```
src/model_version_diff/
├── cli.py          # argparse CLI: `diff` and `weights` subcommands
├── diff.py         # VersionDiff orchestrator + one-line verdict()
├── manifest.py     # manifest loading; resolves {"$file": ...} and bare-path refs
├── config_diff.py  # nested-config diff, semantic categories, severity rules
├── weights.py      # safetensors/npz stats extraction + per-layer drift ranking
├── dataset.py      # fingerprint/schema/distribution-drift comparison
├── behavior.py     # probe-set flip rate, transitions, confidence drift
└── report.py       # Markdown and standalone-HTML rendering
```

See [docs/usage.md](docs/usage.md) for the full guide: manifest schema, fingerprint
format, CI integration, and design notes.

## Development

```bash
pip install -e ".[dev]" 2>/dev/null || pip install -e .
python -m pytest
```

## License

MIT — see [LICENSE](LICENSE). Copyright 2026 Anusha Mukka.
