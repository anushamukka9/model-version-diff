"""Weight statistics: per-layer summaries computed without a model framework.

Two capabilities live here:

1. ``compute_weight_stats(path)`` - read a ``.safetensors`` file (pure-Python
   header parsing, no torch/safetensors dependency), a ``.npz`` file, or a
   PyTorch state dict (``.pth``/``.pt``, needs torch installed) and summarize
   every tensor as ``{"shape", "dtype", "mean", "std", "norm", "min", "max",
   "histogram"}``. BF16 tensors are converted to float32 with a small exact
   bit-level conversion.
2. ``diff_weight_stats(old, new)`` - compare two stats dumps and report
   per-layer deltas, added/removed layers, and the layers that moved most.

Weight stats are deliberately lossy (mean/std/norm/histogram per layer), so
they are safe to check into a model registry, email around, or publish -
unlike the raw weights.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Fixed binning for per-layer weight histograms: 16 equal bins over the
# standardized range [-4, 4] sigma. Fixed edges keep histograms comparable
# across versions, so shape drift is measured on distribution *shape* alone,
# independent of scale shifts (those are captured by mean/std/norm).
HIST_BINS = 16
HIST_MIN, HIST_MAX = -4.0, 4.0
HIST_EDGES = [HIST_MIN + i * (HIST_MAX - HIST_MIN) / HIST_BINS for i in range(HIST_BINS + 1)]

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


def _histogram(flat: np.ndarray) -> Tuple[list, list]:
    """16-bin histogram of standardized values over fixed edges.

    Values are z-scored with the tensor's own mean/std, clipped to
    [-4, 4], and binned. Constant tensors (std == 0) put all mass in the
    central bin. Returns (counts, edges).
    """
    std = float(np.std(flat))
    if std > 0:
        z = np.clip((flat - float(np.mean(flat))) / std, HIST_MIN, HIST_MAX)
    else:
        z = np.zeros_like(flat)
    counts, _ = np.histogram(z, bins=np.asarray(HIST_EDGES))
    return [int(c) for c in counts], HIST_EDGES


def _histogram_drift(h_old: list | None, h_new: list | None) -> float | None:
    """L1 distance between two normalized histograms, in [0, 2].

    Returns None when either side has no histogram (e.g. stats files
    produced before histograms existed), so old dumps keep working.
    """
    if not h_old or not h_new or len(h_old) != len(h_new):
        return None
    s_old, s_new = sum(h_old), sum(h_new)
    if s_old <= 0 or s_new <= 0:
        return None
    return float(sum(abs(a / s_old - b / s_new) for a, b in zip(h_old, h_new)))


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


def _read_torch_state_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load a torch.save'd state dict. torch is an optional dependency."""
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            f"Reading {path.name!r} needs torch installed (pip install torch), "
            "or convert the state dict with examples/convert_torch_to_npz.py"
        ) from exc
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]  # unwrap full checkpoints
    tensors: Dict[str, np.ndarray] = {}
    for name, value in obj.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        tensors[str(name)] = np.asarray(value)
    return tensors


def write_safetensors(path: str | Path, tensors: Dict[str, np.ndarray]) -> None:
    """Write tensors to a minimal .safetensors file (float32/float16/int only).

    Small writer for building test fixtures and converting arrays without a
    torch dependency. Only supports dtypes in SAFETENSORS_DTYPES (BF16 excluded).
    """
    inv = {np.dtype(v): k for k, v in SAFETENSORS_DTYPES.items() if v != "bf16"}
    header: Dict[str, Any] = {}
    offset = 0
    blob = bytearray()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype not in inv:
            raise ValueError(f"dtype {arr.dtype} not supported by the minimal writer")
        data = arr.tobytes()
        header[name] = {
            "dtype": inv[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blob += data
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(blob)


def compute_weight_stats(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Compute per-tensor statistics from a weight file.

    Supported: ``.safetensors`` (pure-Python, no torch), ``.npz``,
    ``.pth``/``.pt`` (PyTorch state dict, needs torch installed).

    Returns ``{"tensors": {name: {"shape", "dtype", "numel", "mean", "std",
    "norm", "min", "max", "histogram", "hist_edges"}}}``. ``histogram`` is a
    16-bin histogram of the tensor's standardized values over fixed edges, so
    distribution-shape drift can be compared across versions.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        tensors = _read_safetensors(path)
    elif suffix == ".npz":
        tensors = _read_npz(path)
    elif suffix in (".pth", ".pt"):
        tensors = _read_torch_state_dict(path)
    else:
        raise ValueError(
            f"Unsupported weight file {path.name!r}: expected .safetensors, .npz, .pth, or .pt"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for name, arr in sorted(tensors.items()):
        flat = arr.astype(np.float64, copy=False).ravel()
        counts, edges = _histogram(flat)
        out[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "numel": int(arr.size),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "norm": float(np.linalg.norm(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "histogram": counts,
            "hist_edges": edges,
        }
    return {"tensors": out}


def load_weight_stats(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load weight stats from a stats JSON file, a .safetensors/.npz file,
    or a PyTorch state dict (.pth/.pt, needs torch)."""
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
    changes in mean, std, and norm - a scale-free way to rank which layers
    moved the most between versions. Each record also carries
    ``histogram_drift``: the L1 distance between the layers' standardized
    histograms (0 = identical distribution shape), which catches changes in
    weight-distribution shape that leave mean/std/norm nearly untouched.
    Stats dumps written before histograms existed simply report
    ``histogram_drift`` as None.
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
        rec["histogram_drift"] = _histogram_drift(o.get("histogram"), n.get("histogram"))
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


__all__ = [
    "compute_weight_stats",
    "load_weight_stats",
    "diff_weight_stats",
    "write_safetensors",
    "HIST_BINS",
    "HIST_EDGES",
]
