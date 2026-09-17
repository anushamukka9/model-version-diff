# Usage Guide

## The version manifest

A manifest is a JSON document describing one model version. Every field except
`model`/`version` is optional — include what you have, and the differ compares
what's present on both sides.

```jsonc
{
  "model": "fraud-detector",
  "version": "v2.0.0",
  // Inline object, or {"$file": "config.json"} to keep the manifest small
  "config": {"hidden_size": 128, "learning_rate": 0.0005, "optimizer": "adamw"},

  // Any of:
  //   - path to a stats JSON (output of `model-version-diff weights`, or hand-built)
  //   - path to a .safetensors file  (parsed in pure Python, no torch needed)
  //   - path to a .npz file
  "weights": "v2_weight_stats.json",

  // Dataset fingerprint descriptor (see below), inline or as a file ref
  "dataset": "v2_dataset.json",

  // Probe predictions: {"probe-id": label} or {"probe-id": {"label": ..., "score": ...}}
  "predictions": "v2_probes.json"
}
```

Paths are resolved relative to the manifest file. `{"$file": "x.json"}` forces a
JSON-file read; a bare string ending in `.safetensors`/`.npz` is parsed as weight
tensors, anything else is read as JSON.

## Weight stats format

`model-version-diff weights model.safetensors --out stats.json` writes:

```jsonc
{"tensors": {"encoder.weight": {"shape": [64, 32], "dtype": "float32",
 "numel": 2048, "mean": 0.001, "std": 0.48, "norm": 21.7, "min": -1.9, "max": 1.8}}}
```

You can generate this with your own tooling instead — the differ only needs
`mean`, `std`, and `norm` per layer. Stats are lossy by design: safe to commit
to a registry, attach to PRs, or publish.

**Drift score.** Each compared layer gets `drift_score = |Δmean|/|mean| +
|Δstd|/|std| + |Δnorm|/|norm|`, so layers are ranked by *relative* movement
regardless of scale. A 0.4 shift in a near-zero-mean head layer outranks noise
in a large-norm embedding — which is exactly the signal you want when asking
"which part of the model actually changed?"

## Dataset fingerprint format

```jsonc
{
  "name": "fraud-train",
  "n_samples": 132000,
  "fingerprint": "sha256:…",   // hash of the raw data, if you have one
  "columns": {
    "amount": {"type": "numeric", "mean": 47.5, "std": 21.0, "min": 0, "max": 1200},
    "label":  {"type": "categorical", "cardinality": 2, "top_values": ["legit", "fraud"]}
  }
}
```

The differ reports volume change, fingerprint match/mismatch, added/removed
columns, and per-column drift: standardized mean shift (in old-σ units) for
numeric columns, cardinality and value-set changes for categoricals.

## Config semantics

Configs are flattened to dotted paths (`training.optimizer.lr`) and each key is
classified by name into **hyperparameter**, **architecture**, **data**,
**infra**, or **other**. Severity rules:

- 🔴 **breaking** — architecture key removed or architecture-defining value
  changed (checkpoints/artifacts likely incompatible).
- 🟡 **significant** — hyperparameter value changed (expect different training
  dynamics) or any key removed.
- ⚪ **minor** — everything else (new keys, infra tweaks).

## Behavior diffing

Probe sets should be **fixed** across versions — the same inputs, re-scored.
The differ reports flip rate, the full flip list (with confidence deltas when
scores are present), label transition counts (`fraud -> legit: 12`), confidence
drift on the probes that *didn't* flip, and probe-set skew (ids only in one
dump).

## CI integration

Fail a release pipeline when behavior regresses beyond a threshold:

```yaml
- run: model-version-diff diff prod/v3.json candidate/v4.json --format json --out diff.json
- run: |
    python - <<'EOF'
    import json, sys
    d = json.load(open("diff.json"))
    if d["behavior_diff"]["flip_rate"] > 0.05:
        sys.exit(f"flip rate {d['behavior_diff']['flip_rate']:.1%} exceeds 5% gate")
    EOF
```

## Design notes

- **No framework at diff time.** Manifests + stats keep the tool installable in
  any CI image with just numpy.
- **Determinism.** All list outputs are sorted; reports are stable across runs,
  so diffs of diffs are meaningful.
- **Honest unknowns.** Missing fingerprints report "not provided", not "match";
  probes present in only one dump are listed as skew rather than silently dropped.
