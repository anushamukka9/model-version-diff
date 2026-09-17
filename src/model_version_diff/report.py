"""Report rendering: Markdown and HTML diff reports."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from .diff import VersionDiff

_SEVERITY_EMOJI = {"breaking": "🔴", "significant": "🟡", "minor": "⚪"}


def render_markdown(diff: VersionDiff) -> str:
    """Render a VersionDiff as a Markdown report."""
    L: list[str] = []
    L.append(f"# Model Version Diff: {diff.model}")
    L.append("")
    L.append(f"**{diff.old_version} → {diff.new_version}**")
    L.append("")
    L.append(f"> {diff.verdict()}")
    L.append("")

    # --- Config ---
    L.append("## ⚙️ Configuration changes")
    L.append("")
    if not diff.config_changes:
        L.append("No configuration changes.")
    else:
        L.append("| Severity | Type | Category | Path | Old | New |")
        L.append("|---|---|---|---|---|---|")
        for c in diff.config_changes:
            emoji = _SEVERITY_EMOJI.get(c.severity, "")
            L.append(
                f"| {emoji} {c.severity} | {c.change_type} | {c.category} | `{c.path}` "
                f"| `{_short(c.old)}` | `{_short(c.new)}` |"
            )
            if c.note:
                L[-1] += f"  <!-- {c.note} -->"
    L.append("")

    # --- Weights ---
    wd = diff.weight_diff or {}
    summary = wd.get("summary", {})
    L.append("## 🧮 Weight statistics")
    L.append("")
    L.append(
        f"Layers: {summary.get('layers_old', 0)} → {summary.get('layers_new', 0)} "
        f"(+{summary.get('added', 0)} added, -{summary.get('removed', 0)} removed, "
        f"{summary.get('changed', 0)} compared)"
    )
    L.append("")
    if wd.get("added"):
        L.append("**Added layers:** " + ", ".join(f"`{n}`" for n in wd["added"]))
        L.append("")
    if wd.get("removed"):
        L.append("**Removed layers:** " + ", ".join(f"`{n}`" for n in wd["removed"]))
        L.append("")
    top = wd.get("top_changed", [])
    if top:
        L.append("**Largest per-layer drift** (relative Δmean + Δstd + Δnorm):")
        L.append("")
        L.append("| Layer | Drift score | Δmean | Δstd | Δnorm | Shape changed |")
        L.append("|---|---|---|---|---|---|")
        for r in top:
            L.append(
                f"| `{r['layer']}` | {r['drift_score']:.4f} | {r['mean_delta']:+.4g} "
                f"| {r['std_delta']:+.4g} | {r['norm_delta']:+.4g} | {r['shape_changed']} |"
            )
        L.append("")

    # --- Dataset ---
    dd = diff.dataset_diff
    if dd is not None:
        L.append("## 📦 Training dataset")
        L.append("")
        L.append(f"**{dd.name_old} → {dd.name_new}** — samples: {dd.n_samples_old:,} → {dd.n_samples_new:,} "
                 f"({dd.sample_delta_pct:+.1f}%)")
        fp = {True: "✅ identical", False: "❌ changed", None: "⚠️ not provided"}.get(dd.fingerprint_match)
        L.append(f"Fingerprint: {fp}")
        L.append("")
        if dd.columns_added:
            L.append("Added columns: " + ", ".join(f"`{c}`" for c in dd.columns_added))
        if dd.columns_removed:
            L.append("Removed columns: " + ", ".join(f"`{c}`" for c in dd.columns_removed))
        num_drifts = [d for d in dd.column_drifts if d.kind == "numeric"]
        if num_drifts:
            L.append("")
            L.append("| Column | Std. mean shift (σ) | Spread change | Range old → new |")
            L.append("|---|---|---|---|")
            for d in num_drifts:
                det = d.detail
                L.append(
                    f"| `{d.column}` | {det['standardized_mean_shift']:+.2f}σ "
                    f"| {det['spread_change_rel']:+.1%} | [{det['min_old']}, {det['max_old']}] → "
                    f"[{det['min_new']}, {det['max_new']}] |"
                )
        cat_drifts = [d for d in dd.column_drifts if d.kind == "categorical"]
        for d in cat_drifts:
            det = d.detail
            if det["cardinality_old"] != det["cardinality_new"] or det["new_values"] or det["dropped_values"]:
                L.append("")
                L.append(f"`{d.column}`: cardinality {det['cardinality_old']} → {det['cardinality_new']}; "
                         f"new values: {det['new_values'] or '—'}; dropped: {det['dropped_values'] or '—'}")
        L.append("")

    # --- Behavior ---
    bd = diff.behavior_diff
    if bd is not None:
        L.append("## 🎯 Behavior on probe set")
        L.append("")
        L.append(
            f"Probes: {bd.n_common} common ({bd.n_probes_old} vs {bd.n_probes_new}). "
            f"**Flips: {bd.n_flips} ({100 * bd.flip_rate:.2f}%)** — agreement {100 * bd.agreement_rate:.2f}%."
        )
        L.append("")
        if bd.transitions:
            L.append("**Label transitions:**")
            L.append("")
            for trans, count in bd.transitions.items():
                L.append(f"- `{trans}`: {count}")
            L.append("")
        if bd.flips:
            L.append("| Probe | Old | New | Δ confidence |")
            L.append("|---|---|---|---|")
            for f in bd.flips[:25]:
                sd = f.as_dict()["score_delta"]
                L.append(f"| `{f.probe_id}` | {f.old_label} | {f.new_label} | {sd if sd is not None else '—'} |")
            if len(bd.flips) > 25:
                L.append(f"| … | | | *+{len(bd.flips) - 25} more* |")
            L.append("")
        cs = bd.confidence_stats
        if cs.get("n_scored_agreements"):
            L.append(
                f"Confidence drift on agreeing probes (n={cs['n_scored_agreements']}): "
                f"mean Δ {cs['mean_delta']:+.4f}, max increase {cs['max_increase']:+.4f}, "
                f"max decrease {cs['max_decrease']:+.4f}."
            )
            L.append("")
        if bd.probes_only_old or bd.probes_only_new:
            L.append(f"Probe-set skew: {len(bd.probes_only_old)} probe(s) only in old, "
                     f"{len(bd.probes_only_new)} only in new.")
            L.append("")

    L.append("---")
    L.append(f"_Generated by model-version-diff on {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_")
    return "\n".join(L) + "\n"


def _short(value, limit: int = 40) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    if text is None:
        return "—"
    return text if len(text) <= limit else text[: limit - 1] + "…"


_CSS = """
body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;max-width:960px;margin:2rem auto;
padding:0 1.5rem;color:#1f2328;line-height:1.55}
h1{border-bottom:2px solid #d0d7de;padding-bottom:.4rem}
h2{margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92rem}
th,td{border:1px solid #d0d7de;padding:.45rem .7rem;text-align:left}
th{background:#f6f8fa}
code{background:#f6f8fa;padding:.1rem .35rem;border-radius:4px;font-size:.88em}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.8rem;font-weight:600}
.breaking{background:#ffebe9;color:#a40e26}.significant{background:#fff8c5;color:#7d4e00}
.minor{background:#eef1f4;color:#59636e}
.verdict{background:#f6f8fa;border-left:4px solid #0969da;padding:.8rem 1rem;font-style:italic}
.meta{color:#59636e;font-size:.85rem}
"""


def render_html(diff: VersionDiff) -> str:
    """Render a VersionDiff as a standalone HTML report."""
    md = render_markdown(diff)
    body_lines: list[str] = []
    in_table = False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            tag = "th" if not in_table else "td"
            if not in_table:
                body_lines.append("<table>")
                in_table = True
            body_lines.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body_lines.append("</table>")
            in_table = False
        if s.startswith("### "):
            body_lines.append(f"<h3>{_inline(s[4:])}</h3>")
        elif s.startswith("## "):
            body_lines.append(f"<h2>{_inline(s[3:])}</h2>")
        elif s.startswith("# "):
            body_lines.append(f"<h1>{_inline(s[2:])}</h1>")
        elif s.startswith("- "):
            body_lines.append(f"<ul><li>{_inline(s[2:])}</li></ul>")
        elif s.startswith("> "):
            body_lines.append(f'<div class="verdict">{_inline(s[2:])}</div>')
        elif s == "---":
            body_lines.append("<hr>")
        elif s.startswith("_") and s.endswith("_") and len(s) > 2:
            body_lines.append(f'<p class="meta">{_inline(s[1:-1])}</p>')
        elif s:
            body_lines.append(f"<p>{_inline(s)}</p>")
    if in_table:
        body_lines.append("</table>")

    title = html.escape(f"Model Version Diff: {diff.model} ({diff.old_version} → {diff.new_version})")
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        + "\n".join(body_lines)
        + "\n</body>\n</html>\n"
    )


def _inline(text: str) -> str:
    """Minimal inline Markdown: `code`, **bold**, and emoji passthrough."""
    import re

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


__all__ = ["render_markdown", "render_html"]
