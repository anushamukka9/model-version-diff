"""Training-dataset fingerprint comparison.

A dataset fingerprint is a compact JSON descriptor — no raw data needed::

    {
      "name": "fraud-train",
      "n_samples": 120000,
      "fingerprint": "sha256:…",          # hash of the raw dataset, if known
      "columns": {
        "amount": {"type": "numeric", "mean": 42.1, "std": 18.3, "min": 0.0, "max": 999.0},
        "label":  {"type": "categorical", "cardinality": 2, "top_values": ["legit", "fraud"]}
      }
    }

``diff_datasets`` reports volume change, fingerprint match/mismatch, schema
changes (added/removed/changed columns), and per-column distribution drift:
standardized mean shift for numeric columns and cardinality/top-value
changes for categorical columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ColumnDrift:
    column: str
    kind: str  # "numeric" | "categorical" | "type_changed"
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "kind": self.kind, **self.detail}


@dataclass
class DatasetDiff:
    name_old: str
    name_new: str
    n_samples_old: int
    n_samples_new: int
    fingerprint_match: bool | None
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    column_drifts: List[ColumnDrift] = field(default_factory=list)

    @property
    def sample_delta_pct(self) -> float:
        if not self.n_samples_old:
            return 0.0
        return 100.0 * (self.n_samples_new - self.n_samples_old) / self.n_samples_old

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name_old": self.name_old,
            "name_new": self.name_new,
            "n_samples_old": self.n_samples_old,
            "n_samples_new": self.n_samples_new,
            "sample_delta_pct": round(self.sample_delta_pct, 2),
            "fingerprint_match": self.fingerprint_match,
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "column_drifts": [d.as_dict() for d in self.column_drifts],
        }


def _numeric_drift(col: str, old: Dict[str, Any], new: Dict[str, Any]) -> ColumnDrift:
    o_mean, n_mean = float(old.get("mean", 0.0)), float(new.get("mean", 0.0))
    o_std = float(old.get("std", 0.0)) or 1e-12
    shift = (n_mean - o_mean) / o_std  # standardized mean shift, in old-std units
    o_std_n, n_std = float(old.get("std", 0.0)), float(new.get("std", 0.0))
    spread = (n_std - o_std_n) / (o_std_n or 1e-12)
    return ColumnDrift(
        column=col,
        kind="numeric",
        detail={
            "mean_old": o_mean,
            "mean_new": n_mean,
            "std_old": o_std_n,
            "std_new": n_std,
            "standardized_mean_shift": round(shift, 4),
            "spread_change_rel": round(spread, 4),
            "min_old": old.get("min"),
            "min_new": new.get("min"),
            "max_old": old.get("max"),
            "max_new": new.get("max"),
        },
    )


def _categorical_drift(col: str, old: Dict[str, Any], new: Dict[str, Any]) -> ColumnDrift:
    o_top = old.get("top_values") or []
    n_top = new.get("top_values") or []
    return ColumnDrift(
        column=col,
        kind="categorical",
        detail={
            "cardinality_old": old.get("cardinality"),
            "cardinality_new": new.get("cardinality"),
            "top_values_old": o_top,
            "top_values_new": n_top,
            "new_values": [v for v in n_top if v not in o_top],
            "dropped_values": [v for v in o_top if v not in n_top],
        },
    )


def diff_datasets(old: Dict[str, Any], new: Dict[str, Any]) -> DatasetDiff:
    """Compare two dataset fingerprint dicts."""
    old = old or {}
    new = new or {}
    old_cols, new_cols = old.get("columns", {}) or {}, new.get("columns", {}) or {}
    old_names, new_names = set(old_cols), set(new_cols)

    fp_old, fp_new = old.get("fingerprint"), new.get("fingerprint")
    fp_match = (fp_old == fp_new) if (fp_old and fp_new) else None

    drifts: List[ColumnDrift] = []
    for col in sorted(old_names & new_names):
        o, n = old_cols[col], new_cols[col]
        o_type, n_type = o.get("type"), n.get("type")
        if o_type != n_type:
            drifts.append(ColumnDrift(col, "type_changed", {"type_old": o_type, "type_new": n_type}))
        elif o_type == "numeric":
            drifts.append(_numeric_drift(col, o, n))
        elif o_type == "categorical":
            drifts.append(_categorical_drift(col, o, n))
        else:
            drifts.append(ColumnDrift(col, o_type or "unknown", {"old": o, "new": n}))

    return DatasetDiff(
        name_old=str(old.get("name", "?")),
        name_new=str(new.get("name", "?")),
        n_samples_old=int(old.get("n_samples", 0) or 0),
        n_samples_new=int(new.get("n_samples", 0) or 0),
        fingerprint_match=fp_match,
        columns_added=sorted(new_names - old_names),
        columns_removed=sorted(old_names - new_names),
        column_drifts=drifts,
    )


__all__ = ["DatasetDiff", "ColumnDrift", "diff_datasets"]
