"""Regression gate.

Compares a new EvalSuite against a baseline. Fails (returns non-zero exit
code when run as a script) if any metric regressed by more than threshold.

This is the piece your CI workflow calls on every PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Regression:
    metric_name: str
    baseline: float
    current: float
    delta: float

    def __str__(self) -> str:
        arrow = "↓" if self.delta < 0 else "↑"
        return (
            f"  {arrow} {self.metric_name:<25} "
            f"baseline={self.baseline:.3f} current={self.current:.3f} "
            f"delta={self.delta:+.3f}"
        )


def compare_suites(
    baseline_path: str | Path,
    current_path: str | Path,
    threshold: float = 0.03,
) -> tuple[list[Regression], list[Regression]]:
    """Return (regressions, improvements).

    A regression is a metric whose score dropped by >= threshold.
    An improvement is a metric that went UP by >= threshold.
    """
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    base_means = baseline.get("metric_means", {})
    curr_means = current.get("metric_means", {})

    regressions: list[Regression] = []
    improvements: list[Regression] = []

    all_metrics = set(base_means.keys()) | set(curr_means.keys())
    for metric in sorted(all_metrics):
        b = base_means.get(metric, 0.0)
        c = curr_means.get(metric, 0.0)
        delta = c - b
        if delta <= -threshold:
            regressions.append(Regression(metric, b, c, delta))
        elif delta >= threshold:
            improvements.append(Regression(metric, b, c, delta))

    return regressions, improvements


def format_report(regressions: list[Regression], improvements: list[Regression]) -> str:
    lines = ["=== Regression Gate Report ==="]
    if regressions:
        lines.append(f"\n❌ {len(regressions)} regression(s):")
        lines.extend(str(r) for r in regressions)
    if improvements:
        lines.append(f"\n✅ {len(improvements)} improvement(s):")
        lines.extend(str(i) for i in improvements)
    if not regressions and not improvements:
        lines.append("\n✓ No significant changes.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EvalForge regression gate.")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON suite")
    parser.add_argument("--current", required=True, help="Path to current JSON suite")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Drop (absolute, 0.0-1.0) that counts as a regression. Default: 0.03",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: write the text report here (e.g., for PR comment bodies).",
    )
    args = parser.parse_args(argv)

    regressions, improvements = compare_suites(
        args.baseline, args.current, threshold=args.threshold
    )
    report = format_report(regressions, improvements)
    print(report)

    if args.output:
        Path(args.output).write_text(report)

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
