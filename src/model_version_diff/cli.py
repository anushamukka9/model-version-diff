"""CLI: diff two model version manifests from the terminal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .diff import VersionDiff, diff_versions
from .report import render_html, render_markdown
from .weights import compute_weight_stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model-version-diff",
        description="Diff two model versions: configs, weight stats, dataset fingerprints, "
        "and probe-set behavior. Render as Markdown, HTML, or JSON.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diff", help="Diff two version manifests")
    d.add_argument("old", help="Path to the old version manifest (JSON)")
    d.add_argument("new", help="Path to the new version manifest (JSON)")
    d.add_argument("--format", choices=["markdown", "html", "json"], default="markdown",
                   help="Report format (default: markdown)")
    d.add_argument("--out", "-o", default=None, help="Write the report to this file (default: stdout)")
    d.add_argument("--top-layers", type=int, default=10,
                   help="How many of the most-drifted layers to show (default: 10)")

    w = sub.add_parser("weights", help="Compute weight stats from a .safetensors or .npz file")
    w.add_argument("file", help="Path to the weight file")
    w.add_argument("--out", "-o", default=None, help="Write stats JSON here (default: stdout)")
    return p


def _cmd_diff(args: argparse.Namespace) -> int:
    diff = diff_versions(args.old, args.new)
    if diff.weight_diff and args.top_layers != 10:
        diff.weight_diff["top_changed"] = diff.weight_diff["changed"][: args.top_layers]

    if args.format == "json":
        report = json.dumps(diff.as_dict(), indent=2, default=str) + "\n"
    elif args.format == "html":
        report = render_html(diff)
    else:
        report = render_markdown(diff)

    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote {args.format} report to {args.out}")
        print(f"Verdict: {diff.verdict()}")
    else:
        sys.stdout.write(report)
    return 0


def _cmd_weights(args: argparse.Namespace) -> int:
    stats = compute_weight_stats(args.file)
    payload = json.dumps(stats, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(payload)
        n = len(stats["tensors"])
        print(f"Wrote stats for {n} tensors to {args.out}")
    else:
        sys.stdout.write(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "diff":
        return _cmd_diff(args)
    if args.command == "weights":
        return _cmd_weights(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
