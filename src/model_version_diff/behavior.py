"""Behavior diffs: how predictions changed on a fixed probe set.

Given two prediction dumps over the same probe inputs — each mapping a probe
id to either a bare label or ``{"label": ..., "score": ...}`` —
``diff_predictions`` computes:

- flip rate (fraction of probes whose predicted label changed),
- the full flip list (probe id, old label, new label, confidence delta),
- per (old -> new) label transition counts,
- confidence-change statistics for the probes that did *not* flip
  (the model may be silently more/less certain even where it agrees),
- probes present in only one of the two dumps (probe-set skew).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Flip:
    probe_id: str
    old_label: Any
    new_label: Any
    old_score: float | None = None
    new_score: float | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "old_label": self.old_label,
            "new_label": self.new_label,
            "old_score": self.old_score,
            "new_score": self.new_score,
            "score_delta": (
                None
                if self.old_score is None or self.new_score is None
                else round(self.new_score - self.old_score, 4)
            ),
        }


@dataclass
class BehaviorDiff:
    n_probes_old: int
    n_probes_new: int
    n_common: int
    n_flips: int
    flips: List[Flip] = field(default_factory=list)
    transitions: Dict[str, int] = field(default_factory=dict)
    probes_only_old: List[str] = field(default_factory=list)
    probes_only_new: List[str] = field(default_factory=list)
    confidence_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def flip_rate(self) -> float:
        return self.n_flips / self.n_common if self.n_common else 0.0

    @property
    def agreement_rate(self) -> float:
        return 1.0 - self.flip_rate

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_probes_old": self.n_probes_old,
            "n_probes_new": self.n_probes_new,
            "n_common": self.n_common,
            "n_flips": self.n_flips,
            "flip_rate": round(self.flip_rate, 4),
            "agreement_rate": round(self.agreement_rate, 4),
            "flips": [f.as_dict() for f in self.flips],
            "transitions": self.transitions,
            "probes_only_old": self.probes_only_old,
            "probes_only_new": self.probes_only_new,
            "confidence_stats": self.confidence_stats,
        }


def _normalize(entry: Any) -> Tuple[Any, float | None]:
    if isinstance(entry, dict):
        return entry.get("label"), entry.get("score")
    return entry, None


def diff_predictions(old: Dict[str, Any], new: Dict[str, Any]) -> BehaviorDiff:
    """Diff two probe-prediction dicts."""
    old = old or {}
    new = new or {}
    old_ids, new_ids = set(old), set(new)
    common = sorted(old_ids & new_ids)

    flips: List[Flip] = []
    transitions: Dict[str, int] = {}
    score_deltas: List[float] = []

    for pid in common:
        o_label, o_score = _normalize(old[pid])
        n_label, n_score = _normalize(new[pid])
        if o_label != n_label:
            flips.append(Flip(pid, o_label, n_label, o_score, n_score))
            key = f"{o_label} -> {n_label}"
            transitions[key] = transitions.get(key, 0) + 1
        elif o_score is not None and n_score is not None:
            score_deltas.append(float(n_score) - float(o_score))

    confidence_stats: Dict[str, Any] = {"n_scored_agreements": len(score_deltas)}
    if score_deltas:
        import statistics

        confidence_stats.update(
            {
                "mean_delta": round(statistics.fmean(score_deltas), 4),
                "max_increase": round(max(score_deltas), 4),
                "max_decrease": round(min(score_deltas), 4),
            }
        )

    return BehaviorDiff(
        n_probes_old=len(old),
        n_probes_new=len(new),
        n_common=len(common),
        n_flips=len(flips),
        flips=flips,
        transitions=dict(sorted(transitions.items(), key=lambda kv: kv[1], reverse=True)),
        probes_only_old=sorted(old_ids - new_ids),
        probes_only_new=sorted(new_ids - old_ids),
        confidence_stats=confidence_stats,
    )


__all__ = ["Flip", "BehaviorDiff", "diff_predictions"]
