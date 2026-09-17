"""Weight statistics: per-layer summaries computed without a model framework.

Two capabilities live here:

1. ``compute_weight_stats(path)`` — read a ``.safetensors`` file (pure-Python
   header parsing, no torch/safetensors dependency) or a ``.npz`` file and
   summarize every tensor as ``{"shape", "dtype", "mean", "std", "norm"}``.
   BF16 tensors are converted to float32 with a small exact bit-level
   conversion.
2. ``diff_weight_stats(old, new)`` — compare two stats dumps and report
   per-layer deltas, added/removed layers, and the layers that moved most.

Weight stats are deliberately lossy (mean/std/norm per layer), so they are
safe to check into a model registry, email around, or publish — unlike the
raw weights.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

SAFETENSORS_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": "bf16",
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def _bf16_to_float32(raw: bytes) -> np.ndarray:
    """Convert a buffer of bfloat16 values to float32 via bit expansion."""
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).copy()


def _read_safetensors(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        fh.seek(0, 2)
        total = fh.tell()
        fh.seek(data_start)
        blob = fh.read(total - data_start)

    tensors: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype {meta['dtype']!r} for {name!r}")
        start, end = meta["data_offsets"]
        start += 0  # offsets are relative to the data section
        raw = blob[start:end]
        shape = tuple(meta["shape"])
        if dtype == "bf16":
            arr = _bf16_to_float32(raw).reshape(shape)
        else:
            arr = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        tensors[name] = arr
    return tensors


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def compute_weight_stats(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Compute per-tensor statistics from a .safetensors or .npz file.

    Returns ``{"tensors": {name: {"shape", "dtype", "mean", "std", "norm", "numel"}}}``.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        tensors = _read_safetensors(path)
    elif suffix == ".npz":
        tensors = _read_npz(path)
    else:
        raise ValueError(f"Unsupported weight file {path.name!r}: expected .safetensors or .npz")

    out: Dict[str, Dict[str, Any]] = {}
    for name, arr in sorted(tensors.items()):
        flat = arr.astype(np.float64, copy=False).ravel()
        out[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "numel": int(arr.size),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "norm": float(np.linalg.norm(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        }
    return {"tensors": out}


def load_weight_stats(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load weight stats from a stats JSON file, a .safetensors file, or .npz."""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return compute_weight_stats(path)


def _layer_map(stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if isinstance(stats, dict) and "tensors" in stats and isinstance(stats["tensors"], dict):
        return stats["tensors"]
    return stats if isinstance(stats, dict) else {}


def diff_weight_stats(
    old: Dict[str, Any], new: Dict[str, Any], top_k: int = 10
) -> Dict[str, Any]:
    """Diff two weight-stats dumps.

    Returns a dict with ``added`` / ``removed`` layer names, a ``changed``
    list of per-layer delta records (sorted by drift score), and aggregate
    ``summary`` counts. The drift score for a layer is the sum of relative
    changes in mean, std, and norm — a scale-free way to rank which layers
    moved the most between versions.
    """
    old_layers, new_layers = _layer_map(old), _layer_map(new)
    old_names, new_names = set(old_layers), set(new_layers)

    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)

    changed = []
    for name in sorted(old_names & new_names):
        o, n = old_layers[name], new_layers[name]
        rec: Dict[str, Any] = {"layer": name, "shape_changed": list(o.get("shape", [])) != list(n.get("shape", []))}
        for key in ("mean", "std", "norm"):
            ov, nv = float(o.get(key, 0.0)), float(n.get(key, 0.0))
            denom = abs(ov) if abs(ov) > 1e-12 else 1e-12
            rec[f"{key}_old"] = ov
            rec[f"{key}_new"] = nv
            rec[f"{key}_delta"] = nv - ov
            rec[f"{key}_rel"] = (nv - ov) / denom
        rec["drift_score"] = abs(rec["mean_rel"]) + abs(rec["std_rel"]) + abs(rec["norm_rel"])
        changed.append(rec)

    changed.sort(key=lambda r: r["drift_score"], reverse=True)
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "top_changed": changed[:top_k],
        "summary": {
            "layers_old": len(old_layers),
            "layers_new": len(new_layers),
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
        },
    }


__all__ = ["compute_weight_stats", "load_weight_stats", "diff_weight_stats"]
