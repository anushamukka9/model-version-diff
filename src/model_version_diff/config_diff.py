"""Config / hyperparameter diffs with semantic typing.

A flat or nested config dict is flattened to dotted paths
(``training.optimizer.lr``), then every path is classified into a semantic
category (hyperparameter, architecture, data, infra, other) by matching
well-known key names. Each difference is typed as added / removed / changed
and given a severity:

- ``breaking``   — an architecture key was removed or its shape-defining
  value changed (old checkpoints likely incompatible).
- ``significant`` — a hyperparameter value changed (training dynamics differ).
- ``minor``      — keys added, infra tweaks, or cosmetic changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Normalized key -> semantic category. Matching is substring-based on the
# last path segment so ``training.learning_rate`` and ``lr`` both hit.
_CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "hyperparameter": (
        "learning_rate", "lr", "batch_size", "batchsize", "epochs", "num_epochs",
        "optimizer", "scheduler", "warmup", "weight_decay", "weightdecay",
        "dropout", "drop_out", "grad_clip", "gradient_clip", "max_grad_norm",
        "seed", "accumulation", "grad_accum", "momentum", "beta1", "beta2",
        "eps", "adam_epsilon", "label_smoothing", "early_stopping",
    ),
    "architecture": (
        "num_layers", "n_layer", "num_hidden_layers", "hidden_size", "hidden_dim",
        "d_model", "num_heads", "num_attention_heads", "n_head", "vocab_size",
        "embedding_dim", "embed_dim", "intermediate_size", "ffn_dim",
        "activation", "rope", "max_seq_len", "max_position", "model_type",
        "dtype", "torch_dtype", "quantization", "num_experts", "tie_word_embeddings",
        "layer_norm", "norm_type", "initializer_range",
    ),
    "data": (
        "dataset", "data_path", "train_path", "eval_path", "tokenizer",
        "tokenizer_name", "max_length", "max_seq_length", "augmentation",
        "sampling", "shuffle", "num_workers", "dataloader", "split",
        "preprocessing", "mixture", "data_mixture",
    ),
    "infra": (
        "device", "gpus", "num_gpus", "precision", "mixed_precision", "fp16",
        "bf16", "distributed", "ddp", "deepspeed", "fsdp", "nodes", "num_nodes",
        "strategy", "checkpointing", "compile",
    ),
}

SEVERITY_ORDER = {"breaking": 0, "significant": 1, "minor": 2}


@dataclass
class ConfigChange:
    path: str
    change_type: str  # "added" | "removed" | "changed"
    category: str     # "hyperparameter" | "architecture" | "data" | "infra" | "other"
    old: Any = None
    new: Any = None
    severity: str = "minor"
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "category": self.category,
            "old": self.old,
            "new": self.new,
            "severity": self.severity,
            "note": self.note,
        }


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, path))
    elif isinstance(obj, (list, tuple)):
        flat[prefix or "<root>"] = list(obj)
    else:
        flat[prefix or "<root>"] = obj
    return flat


def _categorize(path: str) -> str:
    leaf = path.split(".")[-1].lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in leaf for kw in keywords):
            return category
    return "other"


def _severity(change_type: str, category: str) -> Tuple[str, str]:
    if change_type == "removed" and category == "architecture":
        return "breaking", "architecture key removed — old artifacts may be incompatible"
    if change_type == "changed" and category == "architecture":
        return "breaking", "architecture-defining value changed — checkpoints likely incompatible"
    if change_type == "changed" and category == "hyperparameter":
        return "significant", "training hyperparameter changed — dynamics / results may differ"
    if change_type == "removed":
        return "significant", "configuration key removed"
    if change_type == "added" and category == "architecture":
        return "significant", "new architecture key — verify compatibility"
    return "minor", ""


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


def diff_configs(old: dict, new: dict) -> List[ConfigChange]:
    """Diff two (possibly nested) config dicts into typed ConfigChange records."""
    old_flat, new_flat = _flatten(old), _flatten(new)
    changes: List[ConfigChange] = []

    for path in sorted(set(old_flat) | set(new_flat)):
        in_old, in_new = path in old_flat, path in new_flat
        if in_old and not in_new:
            change_type, (category, old_v, new_v) = "removed", (_categorize(path), old_flat[path], None)
        elif in_new and not in_old:
            change_type, (category, old_v, new_v) = "added", (_categorize(path), None, new_flat[path])
        elif not _values_equal(old_flat[path], new_flat[path]):
            change_type, (category, old_v, new_v) = "changed", (_categorize(path), old_flat[path], new_flat[path])
        else:
            continue
        severity, note = _severity(change_type, category)
        changes.append(
            ConfigChange(
                path=path, change_type=change_type, category=category,
                old=old_v, new=new_v, severity=severity, note=note,
            )
        )

    changes.sort(key=lambda c: (SEVERITY_ORDER[c.severity], c.path))
    return changes


__all__ = ["ConfigChange", "diff_configs"]
